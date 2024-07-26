import sys
import os
import argparse
import multiprocessing as mp
import platform
import warnings

import cv2
import mmcv
import torch
import lovely_tensors as lt
lt.monkey_patch()
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmgen.apis import set_random_seed
from mmgen.core import build_metric
from mmgen.datasets import build_dataloader, build_dataset
from mmgen.models import build_model
from mmgen.utils import get_root_logger
from lib.apis import evaluate_3d

import pandas as pd

from lib.models.nerf.provider import NeRFDataset
from lib.models.nerf.diffusion_nerf_adapter import DiffusionNeRFAdapter
from lib.models.nerf.utils import *

_distributed_metrics = ['FID', 'IS', 'FIDKID']

def create_opt():
    # See https://stackoverflow.com/questions/27433316/how-to-get-argparse-to-read-arguments-from-a-file-with-an-option-rather-than-pre
    class LoadFromFile (argparse.Action):
        def __call__ (self, parser, namespace, values, option_string = None):
            with values as f:
                # parse arguments in the file and store them in the target namespace
                parser.parse_args(f.read().split(), namespace)

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=open, action=LoadFromFile, help="specify a file filled with more arguments")
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--prior_text', default=None, help="text prompt for shape prior")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray")
    parser.add_argument('-O2', action='store_true', help="equals --backbone vanilla")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--six_views', action='store_true', help="six_views mode: save the images of the six views")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--test_interval', type=int, default=100, help="test on the test set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', default=None)

    parser.add_argument('--image', default=None, help="image prompt")
    parser.add_argument('--image_config', default=None, help="image config csv")

    parser.add_argument('--known_view_interval', type=int, default=4, help="train default view with RGB loss every & iters, only valid if --image is not None.")

    parser.add_argument('--IF', action='store_true', help="experimental: use DeepFloyd IF as the guidance model for nerf stage")

    parser.add_argument('--guidance', type=str, nargs='*', default=['SD'], help='guidance model')
    parser.add_argument('--guidance_scale', type=float, default=100, help="diffusion model classifier-free guidance scale")

    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--mcubes_resolution', type=int, default=256, help="mcubes resolution for extracting mesh")
    parser.add_argument('--decimate_target', type=int, default=5e4, help="target face number for mesh decimation")

    ## Perp-Neg options
    parser.add_argument('--perpneg', action='store_true', help="use perp_neg")
    parser.add_argument('--negative_w', type=float, default=-2, help="The scale of the weights of negative prompts. A larger value will help to avoid the Janus problem, but may cause flat faces. Vary between 0 to -4, depending on the prompt")
    parser.add_argument('--front_decay_factor', type=float, default=2, help="decay factor for the front prompt")
    parser.add_argument('--side_decay_factor', type=float, default=10, help="decay factor for the side prompt")

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--code_lr_mult', type=float, default=10, help="learning rate multiplication for latent code")
    parser.add_argument('--code_color_lr_mult', type=float, default=10, help="learning rate multiplication for latent color code")
    parser.add_argument('--decoder_lr_mult', type=float, default=1, help="learning rate multiplication for latent color code")
    parser.add_argument('--bg_lr_mult', type=float, default=10, help="learning rate multiplication for latent color code")
    parser.add_argument('--ckpt', type=str, default='latest', help="possible options are ['latest', 'scratch', 'best', 'latest_model']")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--taichi_ray', action='store_true', help="use taichi raymarching")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--latent_iter_ratio', type=float, default=0.2, help="training iters that only use albedo shading")
    parser.add_argument('--albedo_iter_ratio', type=float, default=0, help="training iters that only use albedo shading")
    parser.add_argument('--min_ambient_ratio', type=float, default=0.1, help="minimum ambient ratio to use in lambertian shading")
    parser.add_argument('--textureless_ratio', type=float, default=0.2, help="ratio of textureless shading")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--jitter_center', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's center (camera location)")
    parser.add_argument('--jitter_target', type=float, default=0.2, help="amount of jitter to add to sampled camera pose's target (i.e. 'look-at')")
    parser.add_argument('--jitter_up', type=float, default=0.02, help="amount of jitter to add to sampled camera pose's up-axis (i.e. 'camera roll')")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--grad_clip', type=float, default=-1, help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1, help="clip grad of rgb space grad to this limit, negative value disables it")
    # model options
    parser.add_argument('--bg_radius', type=float, default=7, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_activation', type=str, default='exp', choices=['softplus', 'exp'], help="density activation function")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the density blob")
    parser.add_argument('--blob_radius', type=float, default=0.2, help="control the radius for the density blob")
    # network backbone
    parser.add_argument('--backbone', type=str, default='grid', choices=['grid_tcnn', 'grid', 'vanilla', 'grid_taichi'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # try this if CUDA OOM
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    # rendering resolution in training, increase these for better quality / decrease these if CUDA OOM even if --vram_O enabled.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--coarse_to_fine', action='store_true', help="whether to use coarse-to-fine training")
    parser.add_argument('--ws', type=int, nargs='*', default=[64, 128], help="render width list for NeRF in coarse-to-fine training")
    parser.add_argument('--hs', type=int, nargs='*', default=[64, 128], help="render height list for NeRF in coarse-to-fine training")
    parser.add_argument('--resolution_milestones', type=int, nargs='*', default=[5000], help="render height list for NeRF in coarse-to-fine training")
    parser.add_argument('--recon_loss', action='store_true', help="whether to use reconstruction loss as in MVDream")
    parser.add_argument('--recon_loss_iter', type=int, default=0, help="Enable recon loss after x iterations")
    parser.add_argument('--recon_std_rescale', type=float, default=0.5, help="")
    parser.add_argument('--cfg_anneal', action='store_true', help="whether to use CFG annealing")
    parser.add_argument('--ts_anneal', action='store_true', help="whether to use timestep annealing as in MVDream")

    parser.add_argument('--known_view_scale', type=float, default=1.5, help="multiply --h/w by this for known view rendering")
    parser.add_argument('--known_view_noise_scale', type=float, default=2e-3, help="random camera noise added to rays_o and rays_d")
    parser.add_argument('--batch_size', type=int, default=1, help="images to render per batch using NeRF")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")

    parser.add_argument('--radius_range', type=float, nargs='*', default=[4.5, 5.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[10, 30], help="training camera fovy range")

    parser.add_argument('--default_radius', type=float, default=5, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")

    parser.add_argument('--progressive_view', action='store_true', help="progressively expand view sampling range from default to full")
    parser.add_argument('--progressive_view_init_ratio', type=float, default=0.2, help="initial ratio of final range, used for progressive_view")
    
    parser.add_argument('--progressive_level', action='store_true', help="progressively increase gridencoder's max_level")

    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument('--t_range', type=float, nargs='*', default=[0.02, 0.98], help="stable diffusion time steps range")
    parser.add_argument('--dont_override_stuff',action='store_true', help="Don't override t_range, etc.")


    ### regularizations
    parser.add_argument('--lambda_entropy', type=float, default=1e-3, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale for total variation")
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")

    parser.add_argument('--lambda_mesh_normal', type=float, default=0.5, help="loss scale for mesh normal smoothness")
    parser.add_argument('--lambda_mesh_laplacian', type=float, default=0.5, help="loss scale for mesh laplacian")

    parser.add_argument('--lambda_guidance', type=float, default=1, help="loss scale for SDS")
    parser.add_argument('--lambda_rgb', type=float, default=1000, help="loss scale for RGB")
    parser.add_argument('--lambda_mask', type=float, default=500, help="loss scale for mask (alpha)")
    parser.add_argument('--lambda_normal', type=float, default=0, help="loss scale for normal map")
    parser.add_argument('--lambda_depth', type=float, default=10, help="loss scale for relative depth")
    parser.add_argument('--render_normal', action='store_true', help="whether render normal")
    parser.add_argument('--lambda_2d_normal_smooth', type=float, nargs='*', default=[0], help="loss scale for 2D normal image smoothness")
    parser.add_argument('--lambda_3d_normal_smooth', type=float, default=0, help="loss scale for 3D normal image smoothness")
    parser.add_argument('--lambda_direct3d_prior', type=float, nargs='*', default=[0], help="loss scale for 3D DIRECT-3D prior")
    
    parser.add_argument('--lambda_init_shape_prior', type=float, default=0, help="loss scale for initial shape prior")
    parser.add_argument('--penalize_outside_init_mesh', type=float, default=1.0, help="")
    parser.add_argument('--penalize_dummy', type=float, default=1.0, help="")
    parser.add_argument('--dist_exponent', type=float, default=1.0, help="")
    parser.add_argument('--penalize_outside_factor', type=float, default=2.0, help="")

    # 3D diffusion prior
    parser.add_argument('--randinit', action='store_true', help="Use random initialization")
    parser.add_argument('--z_rot45', action='store_true', help="rotate triplane by 45 degrees around z axis")
    parser.add_argument('--geo_only', action='store_true', help="Only use geometry diffusion prior")
    parser.add_argument('--color_only', action='store_true', help="Only use geometry diffusion prior")
    parser.add_argument('--direct3d_recon_loss', action='store_true', help="Use recon loss in direct3d prior as MVDream")
    parser.add_argument('--direct3d_recon_std_rescale', type=float, default=0.5, help="Use recon loss in direct3d prior as MVDream")
    
    ### debugging options
    parser.add_argument('--save_guidance', action='store_true', help="save images of the per-iteration NeRF renders, added noise, denoised (i.e. guidance), fully-denoised. Useful for debugging, but VERY SLOW and takes lots of memory!")
    parser.add_argument('--save_guidance_interval', type=int, default=10, help="save guidance every X step")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=256, help="GUI width, validation width")
    parser.add_argument('--H', type=int, default=256, help="GUI height, validation height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=20, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    parser.add_argument('--zero123_config', type=str, default='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml', help="config file for zero123")
    parser.add_argument('--zero123_ckpt', type=str, default='pretrained/zero123/zero123-xl.ckpt', help="ckpt for zero123")
    parser.add_argument('--zero123_grad_scale', type=str, default='angle', help="whether to scale the gradients based on 'angle' or 'None'")

    parser.add_argument('--dataset_size_train', type=int, default=100, help="Length of train dataset i.e. # of iterations per epoch")
    parser.add_argument('--dataset_size_valid', type=int, default=8, help="# of frames to render in the turntable video in validation")
    parser.add_argument('--dataset_size_test', type=int, default=100, help="# of frames to render in the turntable video at test time")

    parser.add_argument('--exp_start_iter', type=int, default=None, help="start iter # for experiment, to calculate progressive_view and progressive_level")
    parser.add_argument('--exp_end_iter', type=int, default=None, help="end iter # for experiment, to calculate progressive_view and progressive_level")

    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True
        opt.backbone = 'vanilla'
        opt.progressive_level = True

    if opt.IF:
        if 'SD' in opt.guidance:
            opt.guidance.remove('SD')
            opt.guidance.append('IF')
        opt.latent_iter_ratio = 0 # must not do as_latent

    opt.images, opt.ref_radii, opt.ref_polars, opt.ref_azimuths, opt.zero123_ws = [], [], [], [], []
    opt.default_zero123_w = 1

    opt.exp_start_iter = opt.exp_start_iter or 0
    opt.exp_end_iter = opt.exp_end_iter or opt.iters

    # parameters for image-conditioned generation
    if opt.image is not None or opt.image_config is not None:

        if opt.text is None:
            # use zero123 guidance model when only providing image
            opt.guidance = ['zero123']
            if not opt.dont_override_stuff:
                opt.fovy_range = [opt.default_fovy, opt.default_fovy] # fix fov as zero123 doesn't support changing fov
                opt.guidance_scale = 5
                opt.lambda_3d_normal_smooth = 10
        else:
            # use stable-diffusion when providing both text and image
            opt.guidance = ['SD', 'clip']
            
            if not opt.dont_override_stuff:
                opt.guidance_scale = 10
                opt.t_range = [0.2, 0.6]
                opt.known_view_interval = 2
                opt.lambda_3d_normal_smooth = 20
            opt.bg_radius = -1

        # smoothness
        opt.lambda_entropy = 1
        opt.lambda_orient = 1

        # latent warmup is not needed
        opt.latent_iter_ratio = 0
        if not opt.dont_override_stuff:
            opt.albedo_iter_ratio = 0
            
            # make shape init more stable
            opt.progressive_view = True
            opt.progressive_level = True

        if opt.image is not None:
            opt.images += [opt.image]
            opt.ref_radii += [opt.default_radius]
            opt.ref_polars += [opt.default_polar]
            opt.ref_azimuths += [opt.default_azimuth]
            opt.zero123_ws += [opt.default_zero123_w]

        if opt.image_config is not None:
            # for multiview (zero123)
            conf = pd.read_csv(opt.image_config, skipinitialspace=True)
            opt.images += list(conf.image)
            opt.ref_radii += list(conf.radius)
            opt.ref_polars += list(conf.polar)
            opt.ref_azimuths += list(conf.azimuth)
            opt.zero123_ws += list(conf.zero123_weight)
            if opt.image is None:
                opt.default_radius = opt.ref_radii[0]
                opt.default_polar = opt.ref_polars[0]
                opt.default_azimuth = opt.ref_azimuths[0]
                opt.default_zero123_w = opt.zero123_ws[0]

    # reset to None
    if len(opt.images) == 0:
        opt.images = None

    # record full range for progressive view expansion
    if opt.progressive_view:
        if not opt.dont_override_stuff:
            # disable as they disturb progressive view
            opt.jitter_pose = False
            
        opt.uniform_sphere_rate = 0
        # back up full range
        opt.full_radius_range = opt.radius_range
        opt.full_theta_range = opt.theta_range
        opt.full_phi_range = opt.phi_range
        opt.full_fovy_range = opt.fovy_range

    print(opt)

    if opt.seed is not None:
        seed_everything(int(opt.seed))
    return opt

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test and eval a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument('--local_rank', type=int, default=0)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def main():
    args = parse_args()
    opt = create_opt()
    if opt.seed is not None:
        seed_everything(int(opt.seed))
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    dirname = opt.workspace #os.path.dirname(args.checkpoint)
    os.makedirs(dirname, exist_ok=True)
    ckpt = os.path.basename(args.checkpoint)

    if 'http' in args.checkpoint:
        log_path = None
    else:
        log_name = ckpt.split('.')[0] + '_eval_log' + '.txt'
        log_path = os.path.join(dirname, log_name)

    logger = get_root_logger(log_file=log_path, log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build the model and load checkpoint
    print("CFG: ", cfg.model.unconditional_guidance_scale)
    cfg.dump(opt.workspace + '/diffusion_config.py')
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = DiffusionNeRFAdapter(opt, model).to(device)

    _ = load_checkpoint(model.model, args.checkpoint, map_location='cpu')

    model.eval()
    model.initialize_code_dreamfusion()
    
    if opt.coarse_to_fine:
        print("[INFO] Use coarse-to-fine training: ", opt.ws, opt.hs)
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, 
                                   Hs=opt.hs, Ws=opt.ws, resolution_milestones=opt.resolution_milestones, 
                                   size=opt.dataset_size_train * opt.batch_size).dataloader()
    else:
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    if opt.optim == 'adan':
        from lib.core.optimizer.optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()

    if 'SD' in opt.guidance:
        from lib.models.guidance.sd_utils import StableDiffusion
        guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range, recon_loss=opt.recon_loss, 
        recon_loss_iter=opt.recon_loss_iter, recon_std_rescale=opt.recon_std_rescale, anneal=opt.ts_anneal, cfg_anneal=opt.cfg_anneal)

    if 'IF' in opt.guidance:
        from lib.models.guidance.if_utils import IF
        guidance['IF'] = IF(device, opt.vram_O, opt.t_range, recon_loss=opt.recon_loss, 
        recon_loss_iter=opt.recon_loss_iter, recon_std_rescale=opt.recon_std_rescale)

    if 'zero123' in opt.guidance:
        from lib.models.guidance.zero123_utils import Zero123
        guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)

    if 'clip' in opt.guidance:
        from lib.models.guidance.clip_utils import CLIP
        guidance['clip'] = CLIP(device)

    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)

    trainer.default_view_data = train_loader._data.get_default_view_data()

    if opt.gui:
        from lib.models.nerf.gui import NeRFGUI
        gui = NeRFGUI(opt, trainer, train_loader)
        gui.render()

    else:
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        trainer.train(train_loader, valid_loader, test_loader, max_epoch)

        if opt.save_mesh:
            trainer.save_mesh()

if __name__ == '__main__':
    main()
