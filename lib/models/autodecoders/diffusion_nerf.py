from pickle import TRUE
import torch
import mmcv
import numpy as np
import json
import random

from copy import deepcopy
from torch.nn.parallel.distributed import DistributedDataParallel
from mmgen.models.builder import MODELS, build_module
from mmgen.models.architectures.common import get_module_device
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.transforms as T

from ...core import eval_psnr, rgetattr
from .base_nerf import get_cam_rays
from .multiscene_nerf import MultiSceneNeRF
from ..prompts import FrozenOpenCLIPEmbedder
# from ..SRmodel import SRtrainer, define_SR 

@MODELS.register_module()
class DiffusionNeRF(MultiSceneNeRF):

    def __init__(self,
                 *args,
                 diffusion=dict(type='GaussianDiffusion'),
                 color_diffusion=dict(type='GaussianDiffusion'),
                 diffusion_use_ema=True,
                 freeze_decoder=True,
                 image_cond=False,
                 use_text_cond=True,
                 use_SR=False,
                 merging_SR=False,
                 code_permute=None,
                 code_reshape=None,
                 autocast_dtype=None,
                 EMpose=False,
                 disentangle_code_iter=False,
                 drop_text_rate=0.0,
                 unconditional_guidance_scale=1.,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_SR = use_SR
        self.merging_SR = merging_SR
        if self.disentangle_code:
            assert self.merging_SR is False

        diffusion.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            self.diffusion_ema = deepcopy(self.diffusion)
        
        if self.disentangle_code:
            color_diffusion.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
            self.color_diffusion = build_module(color_diffusion)
            self.diffusion_use_ema = diffusion_use_ema
            if self.diffusion_use_ema:
                self.color_diffusion_ema = deepcopy(self.color_diffusion)

        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
            if self.decoder_use_ema:
                for param in self.decoder_ema.parameters():
                    param.requires_grad = False
        
        if self.use_SR or self.merging_SR:
            json_file_path = 'configs/sr3_128_512.json'
            json_data_lines = []
            with open(json_file_path, 'r') as file:
                for line in file:
                    line = line.split('//')[0].strip()
                    if line:
                        json_data_lines.append(line)
            json_data = '\n'.join(json_data_lines)
            self.SRopt = json.loads(json_data)
            if self.disentangle_code:
                self.SRopt['loss_name'] = 'geo_l_pix'
            self.SRdiffusion = define_SR(self.SRopt)
            self.SRdiffusion_trainer = SRtrainer(self.SRopt, self.SRdiffusion)
            if self.disentangle_code:
                self.SRopt['loss_name'] = 'color_l_pix'
                self.color_SRdiffusion = define_SR(self.SRopt)
                self.color_SRdiffusion_trainer = SRtrainer(self.SRopt, self.color_SRdiffusion)
        
        if self.use_SR:
            for param in self.diffusion.parameters():
                param.requires_grad = False
                if self.diffusion_use_ema:
                    for param in self.diffusion_ema.parameters():
                        param.requires_grad = False
            if self.disentangle_code:
                for param in self.color_diffusion.parameters():
                    param.requires_grad = False
                    if self.diffusion_use_ema:
                        for param in self.color_diffusion_ema.parameters():
                            param.requires_grad = False

        self.image_cond = image_cond
        self.use_text_cond = use_text_cond
        self.drop_text_rate = drop_text_rate
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.disentangle_code_iter = disentangle_code_iter

        self.code_permute = code_permute
        self.code_reshape = code_reshape
        self.code_reshape_inv = [self.code_size[axis] for axis in self.code_permute] if code_permute is not None \
            else self.code_size
        self.code_permute_inv = [self.code_permute.index(axis) for axis in range(len(self.code_permute))] \
            if code_permute is not None else None

        self.autocast_dtype = autocast_dtype
        # if self.autocast_dtype is not None:
        #     self.diffusion.half()
        #     if self.diffusion_use_ema:
        #         self.diffusion_ema.half()
        
        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key)
        
        if self.use_text_cond:
            self.clip_model = FrozenOpenCLIPEmbedder(layer='penultimate')

        self.EMpose = EMpose

    def code_diff_pr(self, code):
        code_diff = code
        if self.code_permute is not None:
            code_diff = code_diff.permute([0] + [axis + 1 for axis in self.code_permute])  # add batch dimension
        if self.code_reshape is not None:
            code_diff = code_diff.reshape(code.size(0), *self.code_reshape)  # add batch dimension
        return code_diff

    def code_diff_pr_inv(self, code_diff):
        code = code_diff
        if self.code_reshape is not None:
            code = code.reshape(code.size(0), *self.code_reshape_inv)
        if self.code_permute_inv is not None:
            code = code.permute([0] + [axis + 1 for axis in self.code_permute_inv])
        return code

    def train_step(self, data, optimizer, running_status=None):
        raise NotImplementedError("We have to remove this part to pass the code check before release.........hmm...")

    def val_uncond(self, data, show_pbar=False, **kwargs):
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        if self.use_SR or self.merging_SR:
            SRdiffusion_trainer = self.SRdiffusion_trainer
            if self.disentangle_code:
                color_SRdiffusion_trainer = self.color_SRdiffusion_trainer
        if self.disentangle_code:
            color_diffusion = self.color_diffusion_ema if self.diffusion_use_ema else self.color_diffusion

        num_batches = len(data['scene_id'])
        noise = data.get('noise', None)
        if noise is None:
            noise = torch.randn(
                (num_batches, *self.code_size), device=get_module_device(self))
        if self.disentangle_code:
            noise_color = data.get('noise_color', None)
            if noise_color is None:
                noise_color = torch.randn(
                    (num_batches, *self.code_size), device=get_module_device(self))
        
        if "test_TPose_imgs" in data:
            raise NotImplementedError("This is no longer used")
        else:
            concat_cond=None

        if self.use_text_cond:
            with torch.no_grad():
                if 'cond_prompt' in data:
                    text_cond = self.clip_model.encode(data['cond_prompt']).detach()
                else:
                    raise NotImplementedError('TODO')
                    text_cond = torch.rand((2,77,1024)).to(code.device)
                
                if 'inference_prompt' in kwargs:
                    temp_text = kwargs.pop('inference_prompt')
                    print("\n[The prompt you are using for this batch] : ", temp_text)

                    bs = len(data['cond_prompt'])
                    temp_prompt = [temp_text] * bs
                    text_cond = self.clip_model.encode(temp_prompt).detach()

                if self.unconditional_guidance_scale!=1.:
                    unconditional_text = [''] * len(data['cond_prompt'])
                    unconditional_conditioning = self.clip_model.encode(unconditional_text).detach()
                else:
                    unconditional_conditioning = None
        else:
            text_cond = None
        
        if 'test_Rot_angle' in data:
            Rot_angle = data['test_Rot_angle'].unsqueeze(2).repeat(1, 1, 1024)
            text_cond = torch.cat((Rot_angle,text_cond),axis = 1)

        if self.disentangle_code:
            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None): # FP16
                code_geo = diffusion(self.code_diff_pr(noise), text_cond = text_cond, concat_cond=concat_cond, return_loss=False, 
                                    unconditional_guidance_scale=self.unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, 
                                    show_pbar=show_pbar, **kwargs)
                code_geo = self.code_diff_pr_inv(code_geo)

                code_color = color_diffusion(self.code_diff_pr(noise_color), text_cond = text_cond, concat_cond=self.code_diff_pr(code_geo), return_loss=False, 
                                    unconditional_guidance_scale=self.unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, 
                                    show_pbar=show_pbar, **kwargs)
                code_color = self.code_diff_pr_inv(code_color)

            code_out = torch.cat((code_geo.unsqueeze(1), code_color.unsqueeze(1)), dim=1)
        else:
            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None): # FP16
                code_out = diffusion(self.code_diff_pr(noise), text_cond = text_cond, concat_cond=concat_cond, return_loss=False, 
                                    unconditional_guidance_scale=self.unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, 
                                    show_pbar=show_pbar, **kwargs)

            assert not isinstance(code_out, list)
            code_out = self.code_diff_pr_inv(code_out)

        
        code_list = code_out if isinstance(code_out, list) else [code_out]
        density_grid_list = []
        density_bitfield_list = []
        for step_id, code in enumerate(code_list):
            n_inverse_steps = self.test_cfg.get('n_inverse_steps', 0)
            if n_inverse_steps > 0 and step_id == (len(code_list) - 1):
                if self.disentangle_code:
                    raise NotImplementedError("disentangle_code do not support n_inverse_steps")
                for param in diffusion.parameters():
                    param.requires_grad = False
                with torch.enable_grad():
                    code_ = self.code_activation.inverse(code).requires_grad_(True)
                    code_optimizer = self.build_optimizer(code_, self.test_cfg)
                    code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)
                    if show_pbar:
                        pbar = mmcv.ProgressBar(n_inverse_steps)
                    for inverse_step_id in range(n_inverse_steps):
                        code_optimizer.zero_grad()
                        code = self.code_activation(code_)
                        loss, log_vars = diffusion(self.code_diff_pr(code), text_cond = text_cond, return_loss=True, cfg=self.test_cfg)
                        loss.backward()
                        code_optimizer.step()
                        if code_scheduler is not None:
                            code_scheduler.step()
                        if show_pbar:
                            pbar.update()
                code = self.code_activation(code_)
            
            if self.use_SR or self.merging_SR:
                if self.disentangle_code:
                    code_geo = code[:,0]
                    code_color = code[:,1]
                    
                    #### geo SR ####
                    SRdiffusion_trainer.set_new_noise_schedule(
                            self.SRopt['model']['beta_schedule']['val'], schedule_phase='val')

                    code_geo = code_geo.reshape(-1,*self.code_size[-3:])

                    img_SR = F.interpolate(code_geo, (self.code_size_tri[-1]), mode="bilinear", align_corners=True)
                    img_SR /= 2.0
                    temp_data={}
                    temp_data['SR'] = img_SR
                    SRdiffusion_trainer.feed_data(temp_data)
                    SRdiffusion_trainer.test(continous=False,splited_tri=True)
                    code_geo = SRdiffusion_trainer.SR
                    code_geo *= 2.0

                    code_geo = code_geo.view(-1,3,code_geo.shape[1],code_geo.shape[2],code_geo.shape[3])

                    SRdiffusion_trainer.set_new_noise_schedule(
                            self.SRopt['model']['beta_schedule']['train'], schedule_phase='train')
                    
                    #### color SR ####
                    color_SRdiffusion_trainer.set_new_noise_schedule(
                            self.SRopt['model']['beta_schedule']['val'], schedule_phase='val')

                    code_color = code_color.reshape(-1,*self.code_size[-3:])

                    img_SR = F.interpolate(code_color, (self.code_size_tri[-1]), mode="bilinear", align_corners=True)
                    img_SR /= 2.0
                    temp_data={}
                    temp_data['SR'] = img_SR
                    color_SRdiffusion_trainer.feed_data(temp_data)
                    color_SRdiffusion_trainer.test(continous=False,splited_tri=True)
                    code_color = color_SRdiffusion_trainer.SR
                    code_color *= 2.0

                    code_color = code_color.view(-1,3,code_color.shape[1],code_color.shape[2],code_color.shape[3])

                    color_SRdiffusion_trainer.set_new_noise_schedule(
                            self.SRopt['model']['beta_schedule']['train'], schedule_phase='train')

                    ## merge two code
                    code = torch.cat((code_geo.unsqueeze(1), code_color.unsqueeze(1)), dim=1)
                    
                else:
                    SRdiffusion_trainer.set_new_noise_schedule(
                            self.SRopt['model']['beta_schedule']['val'], schedule_phase='val')

                    code = code.view(-1,*self.code_size[-3:])

                    img_SR = F.interpolate(code, (self.code_size_tri[-1]), mode="bilinear", align_corners=True)
                    img_SR /= 2.0
                    temp_data={}
                    temp_data['SR'] = img_SR
                    SRdiffusion_trainer.feed_data(temp_data)
                    SRdiffusion_trainer.test(continous=False,splited_tri=True)
                    code = SRdiffusion_trainer.SR
                    code *= 2.0
                    code = code.view(-1,3,code.shape[1],code.shape[2],code.shape[3])

                    SRdiffusion_trainer.set_new_noise_schedule(
                            self.SRopt['model']['beta_schedule']['train'], schedule_phase='train')

            
            code_list[step_id] = code
            density_grid, density_bitfield = self.get_density(decoder, code, cfg=self.test_cfg)
            density_grid_list.append(density_grid)
            density_bitfield_list.append(density_bitfield)
        
        if isinstance(code_out, list):
            return code_list, density_grid_list, density_bitfield_list
        else:
            return code_list[-1], density_grid_list[-1], density_bitfield_list[-1]


    def val_step(self, data, viz_dir=None, viz_dir_guide=None, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        with torch.no_grad():
            if 'code' in data:
                code, density_grid, density_bitfield = self.load_scene(
                    data, load_density=True)
            elif 'cond_imgs' in data:
                cond_mode = self.test_cfg.get('cond_mode', 'guide')
                if cond_mode == 'guide':
                    code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                elif cond_mode == 'optim':
                    code, density_grid, density_bitfield = self.val_optim(data, **kwargs)
                elif cond_mode == 'guide_optim':
                    code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                    if viz_dir_guide is not None and 'test_poses' in data:
                        self.eval_and_viz(
                            data, decoder, code, density_bitfield,
                            viz_dir=viz_dir_guide, cfg=self.test_cfg)
                    code, density_grid, density_bitfield = self.val_optim(
                        data,
                        code_=self.code_activation.inverse(code).requires_grad_(True),
                        density_grid=density_grid,
                        density_bitfield=density_bitfield,
                        **kwargs)
                else:
                    raise AttributeError
            else:
                code, density_grid, density_bitfield = self.val_uncond(data, **kwargs)

            # ==== evaluate reconstruction ====
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code, density_bitfield,
                    viz_dir=viz_dir, cfg=self.test_cfg)
            else:
                log_vars = dict()
                pred_imgs = None
                if viz_dir is None:
                    viz_dir = self.test_cfg.get('viz_dir', None)
                if viz_dir is not None:
                    if isinstance(decoder, DistributedDataParallel):
                        decoder = decoder.module
                    decoder.visualize(
                        code, data['scene_name'],
                        viz_dir, code_range=self.test_cfg.get('clip_range', [-1, 1]))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])
            save_mesh = self.test_cfg.get('save_mesh', False)
            if save_mesh:
                mesh_resolution = self.test_cfg.get('mesh_resolution', 256)
                mesh_threshold = self.test_cfg.get('mesh_threshold', 10)
                self.save_mesh(save_dir, decoder, code, data['scene_name'], mesh_resolution, mesh_threshold)

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict
