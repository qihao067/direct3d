from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
import numpy as np
import igl

from copy import deepcopy
import torchvision.transforms as transforms
from mmgen.models.builder import MODELS
from mmgen.models.architectures.common import get_module_device

from lib.ops.encoding import get_encoder
from .network_grid import MLP
from lib.core import extract_geometry

class DiffusionNeRFAdapter(nn.Module):
    def __init__(self,
                 opt=None, 
                 model=None,
                 num_layers_bg=2,
                 hidden_dim_bg=32,):
        super().__init__()
        self.opt = opt
        self.model = model
        self.cuda_ray = True
        self.randinit = self.opt.randinit
        self.disentangle = model.disentangle_code

        self.world2triplane = torch.tensor(
                                [[-1, 0, 0], 
                                [0, 0, -1], 
                                [0, 1, 0]],
                    dtype=torch.float32).cuda() # [3, 3]
        self.freeze_diffusion()

        # background network
        if self.opt.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=6)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def freeze_diffusion(self, freeze=True):
        for p in self.model.diffusion.parameters():
            p.requires_grad = False if freeze else True

    def initialize_code_dreamfusion(self):
        with torch.no_grad():
            self.get_text_cond()
            with torch.autocast(
                device_type='cuda',
                enabled=self.model.autocast_dtype is not None,
                dtype=getattr(torch, self.model.autocast_dtype) if self.model.autocast_dtype is not None else None): # FP16
                code = self.sample_code().detach()
                # code = torch.randn_like(code) # randinit
                self.density_grid, self.density_bitfield = self.model.get_density(self.model.decoder, code, cfg=self.model.test_cfg)
                self.mean_density = torch.mean(self.density_grid.clamp(min=0))
        if self.disentangle:
            self.code = nn.Parameter(code[:, 0].clone(), requires_grad=True)
            self.code_color = nn.Parameter(code[:, 1].clone(), requires_grad=True)
        else:
            self.code = nn.Parameter(code, requires_grad=True)
        # self.extract_geometry()
        # self.save_mesh(self.opt.workspace, ['mesh_init'])

    def extract_geometry(self):
        """
        for shape regularization
        """
        mesh_resolution = self.model.test_cfg.get('mesh_resolution', 64)
        mesh_threshold = self.model.test_cfg.get('mesh_threshold', 10)
        self.vertices_list = []
        self.triangles_list = []
        for code_single in self.code:
            vertices, triangles = extract_geometry(
                                self.model.decoder,
                                code_single,
                                resolution=mesh_resolution,
                                threshold=mesh_threshold)
            self.vertices_list.append(np.array(vertices).astype(float))
            self.triangles_list.append(np.array(triangles).astype(int))
            
    def save_mesh(self, save_dir, save_names):
        mesh_resolution = self.model.test_cfg.get('mesh_resolution', 128)
        mesh_threshold = self.model.test_cfg.get('mesh_threshold', 10)
        self.model.save_mesh(save_dir, self.model.decoder, self.code, save_names, mesh_resolution, mesh_threshold)

    def get_text_cond(self):
        temp_text = self.opt.prior_text
        bs = 1 # does not support batch size > 1
        temp_prompt = [temp_text] * bs
        self.text_cond = self.model.clip_model.encode(temp_prompt).detach()

        if self.model.unconditional_guidance_scale != 1.:
            unconditional_text = [''] * bs
            self.unconditional_conditioning = self.model.clip_model.encode(unconditional_text).detach()
        else:
            self.unconditional_conditioning = None

    def sample_code(self):
        if self.disentangle:
            noise = torch.randn(
                    (1, *self.model.code_size), device='cuda') 
            noise_color = torch.randn(
                    (1, *self.model.code_size), device='cuda')
            with torch.autocast(
                device_type='cuda',
                enabled=self.model.autocast_dtype is not None,
                dtype=getattr(torch, self.model.autocast_dtype) if self.model.autocast_dtype is not None else None): # FP16
                code_geo = self.model.diffusion(
                                    self.model.code_diff_pr(noise), text_cond=self.text_cond, 
                                    concat_cond=None, return_loss=False, 
                                    unconditional_guidance_scale=self.model.unconditional_guidance_scale, 
                                    unconditional_conditioning=self.unconditional_conditioning, 
                                    show_pbar=False, 
                                    save_intermediates=True if self.randinit else False)

                if self.randinit:
                    code_geo = self.model.code_diff_pr_inv(code_geo[51])#
                else:
                    code_geo = self.model.code_diff_pr_inv(code_geo)

                code_color = self.model.color_diffusion(
                                    self.model.code_diff_pr(noise_color), text_cond=self.text_cond, 
                                    concat_cond=self.model.code_diff_pr(code_geo), return_loss=False, 
                                    unconditional_guidance_scale=self.model.unconditional_guidance_scale, 
                                    unconditional_conditioning=self.unconditional_conditioning, 
                                    show_pbar=False)
                code_color = self.model.code_diff_pr_inv(code_color)

            code = torch.cat((code_geo.unsqueeze(1), code_color.unsqueeze(1)), dim=1)
            
            return code
        else:
            noise = torch.randn(
                    (1, *self.model.code_size), device='cuda') 
            
            with torch.autocast(
                device_type='cuda',
                enabled=self.model.autocast_dtype is not None,
                dtype=getattr(torch, self.model.autocast_dtype) if self.model.autocast_dtype is not None else None): # FP16
                code_out = self.model.diffusion(self.model.code_diff_pr(noise), text_cond = self.text_cond, concat_cond=None, return_loss=False, 
                                    # unconditional_guidance_scale=self.model.unconditional_guidance_scale, 
                                    # unconditional_conditioning=self.unconditional_conditioning, 
                                    show_pbar=False)
            code = self.model.code_diff_pr_inv(code_out)
            return code

    def get_direct3d_sds_loss(self):
        if self.disentangle:
            if self.opt.geo_only:
                loss_diffusion = self.model.diffusion.forward_sds(
                                self.model.code_diff_pr(self.code), concat_cond=None, text_cond=self.text_cond, 
                                unconditional_guidance_scale=self.model.unconditional_guidance_scale,
                                unconditional_conditioning=self.unconditional_conditioning,
                                recon_loss_enabled=self.opt.direct3d_recon_loss, 
                                recon_std_rescale=self.opt.direct3d_recon_std_rescale,
                                cfg=self.model.train_cfg) # only use geo code
                return loss_diffusion
            elif self.opt.color_only:
                loss_diffusion_color = self.model.color_diffusion.forward_sds(
                                self.model.code_diff_pr(self.code_color), concat_cond=self.model.code_diff_pr(self.code), text_cond=self.text_cond, 
                                unconditional_guidance_scale=self.model.unconditional_guidance_scale,
                                unconditional_conditioning=self.unconditional_conditioning,
                                recon_loss_enabled=self.opt.direct3d_recon_loss, 
                                recon_std_rescale=self.opt.direct3d_recon_std_rescale,
                                cfg=self.model.train_cfg) 
                loss_diffusion = loss_diffusion_color
                return loss_diffusion
            else:
                loss_diffusion_geo = self.model.diffusion.forward_sds(
                                self.model.code_diff_pr(self.code), concat_cond=None, text_cond=self.text_cond, 
                                unconditional_guidance_scale=self.model.unconditional_guidance_scale,
                                unconditional_conditioning=self.unconditional_conditioning,
                                recon_loss_enabled=self.opt.direct3d_recon_loss, 
                                recon_std_rescale=self.opt.direct3d_recon_std_rescale,
                                cfg=self.model.train_cfg) # only use geo code
                loss_diffusion_color = self.model.color_diffusion.forward_sds(
                                self.model.code_diff_pr(self.code_color), concat_cond=self.model.code_diff_pr(self.code), text_cond=self.text_cond, 
                                unconditional_guidance_scale=self.model.unconditional_guidance_scale,
                                unconditional_conditioning=self.unconditional_conditioning,
                                recon_loss_enabled=self.opt.direct3d_recon_loss, 
                                recon_std_rescale=self.opt.direct3d_recon_std_rescale,
                                cfg=self.model.train_cfg) 
                loss_diffusion = loss_diffusion_geo + loss_diffusion_color
                return loss_diffusion
        else:
            loss_diffusion = self.model.diffusion.forward_sds(
                            self.model.code_diff_pr(self.code), concat_cond=None, text_cond=self.text_cond, 
                            unconditional_guidance_scale=self.model.unconditional_guidance_scale,
                            unconditional_conditioning=self.unconditional_conditioning,
                            recon_loss_enabled=self.opt.direct3d_recon_loss, 
                            recon_std_rescale=self.opt.direct3d_recon_std_rescale,
                            cfg=self.model.train_cfg)
            return loss_diffusion

    def get_direct3d_prior_loss_trainloss(self):
        if self.disentangle:
            if self.opt.geo_only:
                loss_diffusion, _ = self.model.diffusion(
                                self.model.code_diff_pr(self.code), concat_cond=None, text_cond=self.text_cond, return_loss=True,
                                cfg=self.model.train_cfg) # only use geo code
                return loss_diffusion
            elif self.opt.color_only:
                loss_diffusion_color, _ = self.model.color_diffusion(
                                self.model.code_diff_pr(self.code_color), concat_cond=self.model.code_diff_pr(self.code), text_cond=self.text_cond, return_loss=True,
                                cfg=self.model.train_cfg) 
                loss_diffusion = loss_diffusion_color
                return loss_diffusion
            else:
                loss_diffusion_geo, _ = self.model.diffusion(
                                self.model.code_diff_pr(self.code), concat_cond=None, text_cond=self.text_cond, return_loss=True,
                                cfg=self.model.train_cfg) # only use geo code
                loss_diffusion_color, _ = self.model.color_diffusion(
                                self.model.code_diff_pr(self.code_color), concat_cond=self.model.code_diff_pr(self.code), text_cond=self.text_cond, return_loss=True,
                                cfg=self.model.train_cfg) 
                loss_diffusion = loss_diffusion_geo + loss_diffusion_color
                return loss_diffusion
        else:
            loss_diffusion, _ = self.model.diffusion(
                            self.model.code_diff_pr(self.code), concat_cond=None, text_cond=self.text_cond, return_loss=True,
                            cfg=self.model.train_cfg)
            return loss_diffusion
    
    def get_shape_regularization_loss(self, pts, dirs, sigmas):
        """ Force the points to be inside the mesh
        pts: [B, N, 3]
        dirs: [B, N, 3]
        """
        # subsampling 
        pts = pts[:, ::100, :]
        dirs = dirs[:, ::100, :]
        sigmas = sigmas[:, ::100]
        device = pts.device
        smpl_reg = torch.tensor(0.0, requires_grad=True).float().to(device)
        # print("pts: ", pts)
        can_mesh_verts = self.vertices_list[0] # V, 3, assuming only one mesh
        can_mesh_faces = self.triangles_list[0] # F, 3

        dist_human, _, _ = igl.signed_distance(
            pts.reshape(-1, 3).detach().cpu().numpy(),
            can_mesh_verts,
            can_mesh_faces,
        )
        inside_volume = dist_human < 0
        if inside_volume.sum() > 0:
            smpl_reg = smpl_reg + F.mse_loss(
                1 - torch.exp(-torch.relu(sigmas.reshape(-1)[inside_volume])),
                torch.ones_like(sigmas.reshape(-1)[inside_volume])
            ) * self.opt.penalize_outside_init_mesh

        # generate random samples inside a box in canonical space
        if self.opt.penalize_dummy > 0:
            
            dummy_pts = (torch.rand(pts.shape, dtype=pts.dtype, device=device) - 0.5) * 2 # [-1, 1]
            dummy_sigmas, _, _ = self.model.decoder.point_decode(dummy_pts, dirs, self.code)
            
            dist_dummy, _, _ = igl.signed_distance(
                dummy_pts.reshape(-1, 3).detach().cpu().numpy(),
                can_mesh_verts,
                can_mesh_faces,
            )
            dummy_inside = dist_dummy < 0
            dummy_outside = dist_dummy > 0
            if dummy_inside.sum() > 0:
                smpl_reg = smpl_reg + F.mse_loss(
                    1 - torch.exp(-torch.relu(dummy_sigmas.reshape(-1)[dummy_inside])),
                    torch.ones_like(dummy_sigmas.reshape(-1)[dummy_inside])
                ) * self.opt.penalize_dummy
            if dummy_outside.sum() > 0:
                smpl_reg = smpl_reg + F.l1_loss(
                    (1 - torch.exp(-torch.relu(dummy_sigmas.reshape(-1)[dummy_outside]))) * torch.pow(torch.abs(torch.from_numpy(dist_dummy[dummy_outside]).to(device)) * self.opt.penalize_outside_factor, self.opt.dist_exponent),
                    torch.zeros_like(dummy_sigmas.reshape(-1)[dummy_outside])
                ) * self.opt.penalize_dummy
        return smpl_reg
    
    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs
    
    def render(self, rays_o, rays_d, mvp, h, w, staged=False, max_ray_batch=4096, bg_color=None, render_normal=False, **kwargs):
        # rays_o, rays_d: [B, N, 3]
        # return: pred_rgb: [B, N, 3]
        
        rays_o = torch.matmul(rays_o, self.world2triplane.transpose(0, 1)) # [B, N, 3]
        rays_d = torch.matmul(rays_d, self.world2triplane.transpose(0, 1)) # [B, N, 3]

        decoder = self.model.decoder_ema if self.model.decoder_use_ema else self.model.decoder
        dt_gamma = torch.tensor([0]).type_as(rays_o) # opt.dt_gamma
        
        num_scenes = 1
        max_render_rays = max_ray_batch
        if 0 < max_render_rays < rays_o.size(1):
            rays_o = rays_o.split(max_render_rays, dim=1)
            rays_d = rays_d.split(max_render_rays, dim=1)
        else:
            rays_o = [rays_o]
            rays_d = [rays_d]

        out_image = []
        out_depth = []
        out_normal = []
        out_weights = []
        out_weights_sum = []
        out_xyzs = []
        out_dirs = []
        out_sigmas = []
        loss_normal_perturb = 0
        code = torch.cat((self.code.unsqueeze(1), self.code_color.unsqueeze(1)), dim=1) if self.disentangle else self.code
        for rays_o_single, rays_d_single in zip(rays_o, rays_d):
            outputs = decoder(
                rays_o_single, rays_d_single,
                code, self.density_bitfield, self.model.grid_size,
                dt_gamma=dt_gamma, perturb=False)
            
            # mix background color
            if bg_color is None:
                if self.opt.bg_radius > 0:
                    # use the bg model to calculate bg_color
                    _bg_color = self.background(rays_d_single) # [B, N, 3]
                else:
                    _bg_color = 1
            else:
                _bg_color = bg_color

            weights_sum = torch.stack(outputs['weights_sum'], dim=0) if num_scenes > 1 else outputs['weights_sum'][0]
            if decoder.training:
                weights = torch.stack(outputs['weights'], dim=0) if num_scenes > 1 else outputs['weights'][0]
                xyzs = torch.stack(outputs['xyzs'], dim=0) if num_scenes > 1 else outputs['xyzs'][0]
                dirs = torch.stack(outputs['dirs'], dim=0) if num_scenes > 1 else outputs['dirs'][0]
                sigmas = torch.stack(outputs['sigmas'], dim=0) if num_scenes > 1 else outputs['sigmas'] # output['sigmas'] (num_all_points,)
            rgbs = (torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0]) \
                   + (_bg_color if num_scenes > 1 else _bg_color[0]) * (1 - weights_sum.unsqueeze(-1)) # [max_render_rays, 3]
            
            normals = torch.stack(outputs['normal'], dim=0) if num_scenes > 1 else outputs['normal'][0] if 'normal' in outputs else rgbs
            depth = torch.stack(outputs['depth'], dim=0) if num_scenes > 1 else outputs['depth'][0]
            out_image.append(rgbs[None])
            out_depth.append(depth[None])
            out_normal.append(normals[None])
            if decoder.training:
                out_weights.append(weights[None])
                out_xyzs.append(xyzs[None])
                out_dirs.append(dirs[None])
                out_sigmas.append(sigmas[None])
                
            out_weights_sum.append(weights_sum[None])

            loss_normal_perturb = loss_normal_perturb + (outputs['loss_normal_perturb'] if 'loss_normal_perturb' in outputs else 0)
        
        out_image = torch.cat(out_image, dim=1) if len(out_image) > 1 else out_image[0]
        out_depth = torch.cat(out_depth, dim=1) if len(out_depth) > 1 else out_depth[0]
        out_normal = torch.cat(out_normal, dim=1) if len(out_normal) > 1 else out_normal[0]
        if decoder.training:
            out_weights = torch.cat(out_weights, dim=1) if len(out_weights) > 1 else out_weights[0]
            out_xyzs = torch.cat(out_xyzs, dim=1) if len(out_xyzs) > 1 else out_xyzs[0]
            out_dirs = torch.cat(out_dirs, dim=1) if len(out_dirs) > 1 else out_dirs[0]
            
            out_sigmas = torch.cat(out_sigmas, dim=1) if len(out_sigmas) > 1 else out_sigmas[0]
            
        out_weights_sum = torch.cat(out_weights_sum, dim=1) if len(out_weights_sum) > 1 else out_weights_sum[0]

        results = {}
        results['depth'] = out_depth
        results['image'] = out_image # B x N x 3
        results['weights_sum'] = out_weights_sum
        results['normal_image'] = out_normal
        if decoder.training:
            results['weights'] = out_weights
            results['xyzs'] = out_xyzs
            results['dirs'] = out_dirs
            results['sigmas'] = out_sigmas
            results['loss_normal_perturb'] = loss_normal_perturb
        return results
    
    def get_params(self, lr):
        params = [
            {'params': self.code, 'lr': lr * self.opt.code_lr_mult},
            {'params': self.model.decoder.parameters(), 'lr': lr * self.opt.decoder_lr_mult},
        ]
        if self.opt.bg_radius > 0:
            params.append({'params': self.bg_net.parameters(), 'lr': lr * self.opt.bg_lr_mult})
        if self.disentangle:
            params.append({'params': self.code_color, 'lr': lr * self.opt.code_color_lr_mult})
        return params

    @torch.no_grad()
    def export_mesh(self, path, resolution=None, decimate_target=-1, S=128):
        
        mesh_resolution = resolution if resolution is not None else 256
        mesh_threshold = self.model.test_cfg.get('mesh_threshold', 10)
        mesh_threshold = 10
        self.model.save_mesh(path, self.model.decoder, self.code, ['scene'], mesh_resolution, mesh_threshold)

    def update_extra_state(self, decay=0.95, S=128):
        self.model.update_extra_state(self.model.decoder, self.code, self.density_grid, self.density_bitfield,
                           0, density_thresh=self.model.test_cfg.get('density_thresh', 0.01), decay=decay, S=S)
        self.mean_density = torch.mean(self.density_grid.clamp(min=0))