import numpy as np
import time

import torch
import torch.nn as nn
from itertools import groupby
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _raymarching as _backend
except ImportError:
    from .backend import _backend


# ----------------------------------------
# utils
# ----------------------------------------

class _near_far_from_aabb(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, aabb, min_near=0.2):
        ''' near_far_from_aabb, CUDA implementation
        Calculate rays' intersection time (near and far) with aabb
        Args:
            rays_o: float, [N, 3]
            rays_d: float, [N, 3]
            aabb: float, [6], (xmin, ymin, zmin, xmax, ymax, zmax)
            min_near: float, scalar
        Returns:
            nears: float, [N]
            fars: float, [N]
        '''
        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda:
            rays_d = rays_d.cuda()
        if not aabb.is_cuda:
            aabb = aabb.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # num rays

        nears = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)
        fars = torch.empty(N, dtype=rays_o.dtype, device=rays_o.device)

        _backend.near_far_from_aabb(rays_o, rays_d, aabb, N, min_near, nears, fars)

        return nears, fars


near_far_from_aabb = _near_far_from_aabb.apply


def batch_near_far_from_aabb(rays_o, rays_d, aabb, min_near=0.2):

    if isinstance(rays_o, torch.Tensor):
        assert rays_o.size() == rays_d.size()
        num_scenes, num_rays, _ = rays_o.size()
        nears, fars = near_far_from_aabb(
            rays_o.reshape(num_scenes * num_rays, 3), rays_d.reshape(num_scenes * num_rays, 3),
            aabb, min_near)
        nears = nears.reshape(num_scenes, num_rays)
        fars = fars.reshape(num_scenes, num_rays)

    else:
        if len(rays_o) == 1:
            nears, fars = near_far_from_aabb(
                rays_o[0], rays_d[0], aabb, min_near)
            nears = [nears]
            fars = [fars]
        else:
            num_rays_per_scene = [rays_o_single.size(0) for rays_o_single in rays_o]
            nears, fars = near_far_from_aabb(
                torch.cat(rays_o, dim=0), torch.cat(rays_d, dim=0), aabb, min_near)
            nears = nears.split(num_rays_per_scene)
            fars = fars.split(num_rays_per_scene)

    return nears, fars


class _sph_from_ray(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, radius):
        ''' sph_from_ray, CUDA implementation
        get spherical coordinate on the background sphere from rays.
        Assume rays_o are inside the Sphere(radius).
        Args:
            rays_o: [N, 3]
            rays_d: [N, 3]
            radius: scalar, float
        Return:
            coords: [N, 2], in [-1, 1], theta and phi on a sphere. (further-surface)
        '''
        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # num rays

        coords = torch.empty(N, 2, dtype=rays_o.dtype, device=rays_o.device)

        _backend.sph_from_ray(rays_o, rays_d, radius, N, coords)

        return coords


sph_from_ray = _sph_from_ray.apply


class _morton3D(Function):
    @staticmethod
    def forward(ctx, coords):
        ''' morton3D, CUDA implementation
        Args:
            coords: [N, 3], int32, in [0, 128) (for some reason there is no uint32 tensor in torch...) 
            TODO: check if the coord range is valid! (current 128 is safe)
        Returns:
            indices: [N], int32, in [0, 128^3)
            
        '''
        if not coords.is_cuda: coords = coords.cuda()

        N = coords.shape[0]

        indices = torch.empty(N, dtype=torch.int32, device=coords.device)

        _backend.morton3D(coords.int(), N, indices)

        return indices


morton3D = _morton3D.apply


class _morton3D_invert(Function):
    @staticmethod
    def forward(ctx, indices):
        ''' morton3D_invert, CUDA implementation
        Args:
            indices: [N], int32, in [0, 128^3)
        Returns:
            coords: [N, 3], int32, in [0, 128)
            
        '''
        if not indices.is_cuda: indices = indices.cuda()

        N = indices.shape[0]

        coords = torch.empty(N, 3, dtype=torch.int32, device=indices.device)

        _backend.morton3D_invert(indices.int(), N, coords)

        return coords


morton3D_invert = _morton3D_invert.apply


class _packbits(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grid, thresh, bitfield=None):
        ''' packbits, CUDA implementation
        Pack up the density grid into a bit field to accelerate ray marching.
        Args:
            grid: float, [C, H * H * H], assume H % 2 == 0
            thresh: float, threshold
        Returns:
            bitfield: uint8, [C, H * H * H / 8]
        '''
        if not grid.is_cuda: grid = grid.cuda()
        grid = grid.contiguous()

        C = grid.shape[0]
        H3 = grid.shape[1]
        N = C * H3 // 8

        if bitfield is None:
            bitfield = torch.empty(N, dtype=torch.uint8, device=grid.device)

        _backend.packbits(grid, N, thresh, bitfield)

        return bitfield


packbits = _packbits.apply


# ----------------------------------------
# train functions
# ----------------------------------------

class _march_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter=None, mean_count=-1,
                perturb=False, align=-1, force_all_rays=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only)
        Args:
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            step_counter: int32, (2), used to count the actual number of generated points.
            mean_count: int32, estimated mean steps to accelerate training. (but will randomly drop rays if the actual point count exceeded this threshold.)
            perturb: bool
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            force_all_rays: bool, ignore step_counter and mean_count, always calculate all rays. Useful if rendering the whole image, instead of some rays.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [M, 3], all generated points' coords. (all rays concated, need to use `rays` to extract points belonging to each ray)
            dirs: float, [M, 3], all generated points' view dirs.
            deltas: float, [M, 2], all generated points' deltas. (first for RGB, second for Depth)
            rays: int32, [N, 3], all rays' (index, point_offset, point_count), e.g., xyzs[rays[i, 1]:rays[i, 1] + rays[i, 2]] --> points belonging to rays[i, 0]
        '''

        if not rays_o.is_cuda:
            rays_o = rays_o.cuda()
        if not rays_d.is_cuda:
            rays_d = rays_d.cuda()
        if not density_bitfield.is_cuda:
            density_bitfield = density_bitfield.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        density_bitfield = density_bitfield.contiguous()

        N = rays_o.shape[0]  # num rays
        M = N * max_steps  # init max points number in total

        # running average based on previous epoch (mimic `measured_batch_size_before_compaction` in instant-ngp)
        # It estimate the max points number to enable faster training, but will lead to random ignored rays if underestimated.
        if not force_all_rays and mean_count > 0:
            if align > 0:
                mean_count += align - mean_count % align
            M = mean_count

        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)
        rays = torch.empty(N, 3, dtype=torch.int32, device=rays_o.device)  # id, offset, num_steps

        if step_counter is None:
            step_counter = torch.zeros(2, dtype=torch.int32, device=rays_o.device)  # point counter, ray counter

        if perturb:
            noises = torch.rand(N, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(N, dtype=rays_o.dtype, device=rays_o.device)

        _backend.march_rays_train(rays_o, rays_d, density_bitfield, bound, dt_gamma, max_steps, N, C, H, M, nears, fars,
                                  xyzs, dirs, deltas, rays, step_counter,
                                  noises)  # m is the actually used points number

        # print(step_counter, M)

        # only used at the first (few) epochs.
        if force_all_rays or mean_count <= 0:
            m = step_counter[0].item()  # D2H copy
            if align > 0:
                m += align - m % align
            xyzs = xyzs[:m]
            dirs = dirs[:m]
            deltas = deltas[:m]

            # torch.cuda.empty_cache()
        return xyzs, dirs, deltas, rays


def march_rays_train(
        rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter=None, mean_count=-1,
        perturb=False, align=-1, force_all_rays=False, dt_gamma=0, max_steps=1024):
    return _march_rays_train.apply(
        rays_o, rays_d, bound, density_bitfield, C, H, nears, fars, step_counter, mean_count,
        perturb, align, force_all_rays, dt_gamma, max_steps)


class _composite_rays_train(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, rays, T_thresh=1e-4):
        """ composite rays' rgbs, according to the ray marching formula.
        Args:
            rgbs: float, [M, 3]
            sigmas: float, [M,]
            deltas: float, [M, 2]
            rays: int32, [N, 3]
        Returns:
            weights: float, [M]
            weights_sum: float, [N,], the alpha channel
            depth: float, [N, ], the Depth
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        """

        sigmas = sigmas.contiguous()
        rgbs = rgbs.contiguous()

        M = sigmas.shape[0]
        N = rays.shape[0]

        weights = torch.zeros(M, dtype=sigmas.dtype, device=sigmas.device) # may leave unmodified, so init with 0
        weights_sum = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)

        depth = torch.empty(N, dtype=sigmas.dtype, device=sigmas.device)
        image = torch.empty(N, 3, dtype=sigmas.dtype, device=sigmas.device)


        _backend.composite_rays_train_forward(sigmas, rgbs, deltas, rays, M, N, T_thresh, weights, weights_sum, depth, image)
        # _backend.composite_rays_train_forward(sigmas, rgbs, deltas, rays, M, N, T_thresh, weights, weights_sum, depth, image)

        ctx.save_for_backward(sigmas, rgbs, deltas, rays, weights, weights_sum, depth, image)
        ctx.dims = [M, N, T_thresh]

        return weights, weights_sum, depth, image

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_weights, grad_weights_sum, grad_depth, grad_image):
        # NOTE: grad_depth is not used now! It won't be propagated to sigmas.

        grad_weights = grad_weights.contiguous()
        grad_weights_sum = grad_weights_sum.contiguous()
        grad_image = grad_image.contiguous()

        sigmas, rgbs, deltas, rays, weights, weights_sum, depth, image = ctx.saved_tensors
        M, N, T_thresh = ctx.dims

        grad_sigmas = torch.zeros_like(sigmas)
        grad_rgbs = torch.zeros_like(rgbs)

        _backend.composite_rays_train_backward(grad_weights, grad_weights_sum, grad_image, sigmas, rgbs, deltas, rays, weights_sum,
                                               image, M, N, T_thresh, grad_sigmas, grad_rgbs)

        return grad_sigmas, grad_rgbs, None, None, None


composite_rays_train = _composite_rays_train.apply


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def batch_composite_rays_train(sigmas, rgbs, deltas, rays, num_points, T_thresh=1e-4):
    """
    Args:
        sigmas: (num_all_points,)
        rgbs: (num_all_points, 3)
        deltas: (num_scenes, (num_rays_per_scene, 2))
        rays: (num_scenes, (num_rays_per_scene, 3)) in [index, point_offset, point_count]
        num_points: (num_scenes,)
    """
    num_scenes = len(deltas)

    if num_scenes > 1:
        deltas_ = torch.cat(deltas, dim=0)

        rays_ = []
        num_rays = []
        ray_offset_total = 0
        point_offset_total = 0
        for ray_single, num_points_per_scene in zip(rays, num_points):
            rays_.append(
                torch.stack(
                    [ray_single[:, 0] + ray_offset_total,
                     ray_single[:, 1] + point_offset_total,
                     ray_single[:, 2]], dim=-1))
            num_rays_per_scene = ray_single.size(0)
            ray_offset_total += num_rays_per_scene
            point_offset_total += num_points_per_scene
            num_rays.append(num_rays_per_scene)
        rays_ = torch.cat(rays_, dim=0)

        weights_, weights_sum_, depth_, image_ = composite_rays_train(sigmas, rgbs, deltas_, rays_, T_thresh)
        if all_equal(num_rays):
            weights = weights_.reshape(num_scenes, -1)
            weights_sum = weights_sum_.reshape(num_scenes, num_rays[0])
            depth = depth_.reshape(num_scenes, num_rays[0])
            image = image_.reshape(num_scenes, num_rays[0], 3)
        else:
            raise NotImplementedError("Weight should have equal ray number")
            weights_sum = weights_sum_.split(num_rays, dim=0)
            depth = depth_.split(num_rays, dim=0)
            image = image_.split(num_rays, dim=0)

    else:
        weights, weights_sum, depth, image = composite_rays_train(sigmas, rgbs, deltas[0], rays[0], T_thresh)
        weights = weights[None]
        weights_sum = weights_sum[None]
        depth = depth[None]
        image = image[None]

    return weights, weights_sum, depth, image


# ----------------------------------------
# infer functions
# ----------------------------------------

class _march_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_bitfield, C, H, near, far,
                align=-1, perturb=False, dt_gamma=0, max_steps=1024):
        ''' march rays to generate points (forward only, for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [N], the alive rays' IDs in N (N >= n_alive, but we only use first n_alive)
            rays_t: float, [N], the alive rays' time, we only use the first n_alive.
            rays_o/d: float, [N, 3]
            bound: float, scalar
            density_bitfield: uint8: [CHHH // 8]
            C: int
            H: int
            nears/fars: float, [N]
            align: int, pad output so its size is dividable by align, set to -1 to disable.
            perturb: bool/int, int > 0 is used as the random seed.
            dt_gamma: float, called cone_angle in instant-ngp, exponentially accelerate ray marching if > 0. (very significant effect, but generally lead to worse performance)
            max_steps: int, max number of sampled points along each ray, also affect min_stepsize.
        Returns:
            xyzs: float, [n_alive * n_step, 3], all generated points' coords
            dirs: float, [n_alive * n_step, 3], all generated points' view dirs.
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        '''

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        M = n_alive * n_step

        if align > 0:
            M += align - (M % align)

        xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device)
        deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device)  # 2 vals, one for rgb, one for depth

        if perturb:
            # torch.manual_seed(perturb) # test_gui uses spp index as seed
            noises = torch.rand(n_alive, dtype=rays_o.dtype, device=rays_o.device)
        else:
            noises = torch.zeros(n_alive, dtype=rays_o.dtype, device=rays_o.device)

        _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, dt_gamma, max_steps, C, H,
                            density_bitfield, near, far, xyzs, dirs, deltas, noises)

        return xyzs, dirs, deltas


def march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_bitfield, C, H, near, far,
               align=-1, perturb=False, dt_gamma=0, max_steps=1024):
    return _march_rays.apply(
        n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, density_bitfield, C, H, near, far,
        align, perturb, dt_gamma, max_steps)


class _composite_rays(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # need to cast sigmas & rgbs to float
    def forward(ctx, n_alive, n_step, rays_alive,
                rays_t, sigmas, rgbs, deltas, weights_sum, depth, image,
                T_thresh=1e-2):
        """ composite rays' rgbs, according to the ray marching formula. (for inference)
        Args:
            n_alive: int, number of alive rays
            n_step: int, how many steps we march
            rays_alive: int, [n_alive], the alive rays' IDs in N (N >= n_alive)
            rays_t: float, [N], the alive rays' time
            sigmas: float, [n_alive * n_step,]
            rgbs: float, [n_alive * n_step, 3]
            deltas: float, [n_alive * n_step, 2], all generated points' deltas (here we record two deltas, the first is for RGB, the second for depth).
        In-place Outputs:
            weights_sum: float, [N,], the alpha channel
            depth: float, [N,], the depth value
            image: float, [N, 3], the RGB channel (after multiplying alpha!)
        """
        _backend.composite_rays(n_alive, n_step, T_thresh, rays_alive,
                                rays_t, sigmas, rgbs, deltas, weights_sum, depth,
                                image)
        return tuple()


composite_rays = _composite_rays.apply
