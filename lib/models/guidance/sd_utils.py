from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def C(value: list, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98], recon_loss=False, recon_loss_iter=5000, recon_std_rescale=0.5, anneal=False, cfg_anneal=False):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.recon_loss = recon_loss
        self.recon_loss_enabled = False
        self.recon_loss_iter = recon_loss_iter
        self.recon_std_rescale = recon_std_rescale
        self.anneal = anneal
        self.cfg_anneal = cfg_anneal
        self.n_view = 1 # for future MVDream use
        self.min_step_percent = [0, 0.98, 0.02, 8000]
        self.max_step_percent = [0, 0.98, 0.50, 8000]
        # self.min_step_percent = [0, 0.5, 0.02, 8000]
        # self.max_step_percent = [0, 0.5, 0.50, 8000]
        self.cfg_overrides = [5000, 50, 20, 8000]
        self.cfg_override = None

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def predict_start_from_noise(self, latents_noisy, t, noise_pred):
        sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas - 1)

        pred_x0 = extract_into_tensor(sqrt_recip_alphas_cumprod, t, latents_noisy.shape) * latents_noisy - extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, noise_pred.shape) * noise_pred
        # alphas = self.scheduler.alphas.to(latents_noisy.device)
        # total_timesteps = self.max_step - self.min_step + 1
        # index = total_timesteps - t.to(latents_noisy.device) - 1 
        # b = len(noise_pred)
        # a_t = alphas[index].reshape(b,1,1,1).to(self.device)
        # sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        # sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
        # pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
        return pred_x0

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        
        if self.recon_loss:
            # print("[INFO]: Use reconstruction loss from MVDream")
            # reconstruct x0
            latents_recon = self.predict_start_from_noise(latents_noisy, t, noise_pred)

            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.predict_start_from_noise(latents_noisy, t, noise_pred_pos)
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # # rescale cfg
            # factor = (noise_pred_pos.std([0,1,2,3], keepdim=True) + 1e-8) / (noise_pred.std([0,1,2,3], keepdim=True) + 1e-8)
            # noise_pred_adjust = noise_pred.clone() * factor
            # noise_pred = self.recon_std_rescale * noise_pred_adjust + (1 - self.recon_std_rescale) * noise_pred
            # w(t), sigma_t^2
            w = (1 - self.alphas[t])
            grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            if save_guidance_path:
                with torch.no_grad():
                    if as_latent:
                        pred_rgb_512 = self.decode_latents(latents)

                    # visualize predicted denoised image
                    # The following block of code is equivalent to `predict_start_from_noise`...
                    # see zero123_utils.py's version for a simpler implementation.
                    alphas = self.scheduler.alphas.to(latents)
                    total_timesteps = self.max_step - self.min_step + 1
                    index = total_timesteps - t.to(latents.device) - 1 
                    b = len(noise_pred)
                    a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                    sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                    pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                    result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                    # visualize noisier image
                    result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                    # TODO: also denoise all-the-way

                    # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                    viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                    save_image(viz_images, save_guidance_path)

            targets = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss
    

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                   save_guidance_path:Path=None):
        if self.cfg_override is not None:
            guidance_scale = self.cfg_override

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts       

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
        delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
        noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)            

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        if self.recon_loss_enabled:
            # import ipdb; ipdb.set_trace()
            # print("[INFO]: Use reconstruction loss from MVDream")
            # reconstruct x0
            latents_recon = self.predict_start_from_noise(latents_noisy, t, noise_pred)
            # # debug
            # result_hopefully_less_noisy_image = self.decode_latents(latents_recon.to(latents.type(self.precision_t)))
            # import numpy as np
            # np.save("debug.npy", result_hopefully_less_noisy_image.detach().cpu().numpy())

            # clip or rescale x0
            if self.recon_std_rescale > 0:
                latents_recon_nocfg = self.predict_start_from_noise(latents_noisy, t, noise_pred_text[B:2*B])
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(-1,self.n_view, *latents_recon_nocfg.shape[1:])
                latents_recon_reshape = latents_recon.view(-1,self.n_view, *latents_recon.shape[1:])
                factor = (latents_recon_nocfg_reshape.std([1,2,3,4],keepdim=True) + 1e-8) / (latents_recon_reshape.std([1,2,3,4],keepdim=True) + 1e-8)
                
                latents_recon_adjust = latents_recon.clone() * factor.squeeze(1).repeat_interleave(self.n_view, dim=0)
                latents_recon = self.recon_std_rescale * latents_recon_adjust + (1-self.recon_std_rescale) * latents_recon
                

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = 0.5 * grad_scale * F.mse_loss(latents, latents_recon.detach(), reduction="sum") / latents.shape[0]
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # factor = (noise_pred_uncond.std([0,1,2,3], keepdim=True) + 1e-8) / (noise_pred.std([0,1,2,3], keepdim=True) + 1e-8)
            # noise_pred_adjust = noise_pred.clone() * factor
            # noise_pred = self.recon_std_rescale * noise_pred_adjust + (1 - self.recon_std_rescale) * noise_pred            
            # w(t), sigma_t^2
            w = (1 - self.alphas[t])
            grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            if save_guidance_path:
                with torch.no_grad():
                    if as_latent:
                        pred_rgb_512 = self.decode_latents(latents)

                    # visualize predicted denoised image
                    # The following block of code is equivalent to `predict_start_from_noise`...
                    # see zero123_utils.py's version for a simpler implementation.
                    alphas = self.scheduler.alphas.to(latents)
                    total_timesteps = self.max_step - self.min_step + 1
                    index = total_timesteps - t.to(latents.device) - 1 
                    b = len(noise_pred)
                    a_t = alphas[index].reshape(b,1,1,1).to(self.device)
                    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                    sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b,1,1,1)).to(self.device)                
                    pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt() # current prediction for x_0
                    result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                    # visualize noisier image
                    result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))



                    # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                    viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image],dim=0)
                    save_image(viz_images, save_guidance_path)

            targets = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if self.recon_loss and global_step > self.recon_loss_iter:
            self.recon_loss_enabled = True
            # print("[INFO]: Enable reconstruction loss from MVDream")
        if self.anneal:
            min_step_percent = C(self.min_step_percent, epoch, global_step)
            max_step_percent = C(self.max_step_percent, epoch, global_step)
            self.min_step = int( self.num_train_timesteps * min_step_percent )
            self.max_step = int( self.num_train_timesteps * max_step_percent )

        if self.cfg_anneal:
            self.cfg_override = C(self.cfg_overrides, epoch, global_step)
            

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




