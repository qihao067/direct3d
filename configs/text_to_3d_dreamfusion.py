name = 'text_to_3d_dreamfusion'

model = dict(
    type='DiffusionNeRF',
    code_size=(3, 6, 128, 128),
    code_reshape=(18, 128, 128),
    code_size_tri=(3, 6, 128, 128),
    disentangle_code=True,
    disentangle_code_iter=False,
    diffusion_use_ema=False,
    code_activation=dict(
        type='NormalizedTanhCode', mean=0.0, std=0.5, clip_range=2),
    grid_size=64,
    diffusion=dict(
        type='GaussianDiffusion',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        denoising=dict(
            type='DenoisingUnetMod',
            image_size=128,
            in_channels=18,
            base_channels=128,
            channels_cfg=[1, 2, 2, 4, 4],
            resblocks_per_downsample=2,
            dropout=0.0,
            use_scale_shift_norm=True,
            use_text_condition=True,
            concat_cond_channels=0,
            use_checkpoint=True,
            downsample_conv=True,
            upsample_conv=True,
            num_heads=4,
            attention_res=[32, 16, 8]),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=20,
            )),
    color_diffusion=dict(
        type='GaussianDiffusion',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        denoising=dict(
            type='DenoisingUnetMod',
            image_size=128,
            in_channels=18,
            base_channels=128,
            channels_cfg=[1, 2, 2, 4, 4],
            resblocks_per_downsample=2,
            dropout=0.0,
            use_scale_shift_norm=True,
            use_text_condition=True,
            concat_cond_channels=18,
            use_checkpoint=True,
            downsample_conv=True,
            upsample_conv=True,
            num_heads=4,
            attention_res=[32, 16, 8]),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='lossColor_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=20,
            )),
    decoder=dict(
        type='TriPlaneDecoder',
        interp_mode='bilinear',
        base_layers=[6 * 3, 64, 64, 64],
        density_layers=[64, 1],
        color_layers=[64, 3],
        use_dir_enc=True,
        dir_layers=[16, 64],
        activation='silu',
        sigma_activation='trunc_exp',
        sigmoid_saturation=0.001,
        max_steps=256),
    decoder_use_ema=False,
    freeze_decoder=False,
    use_text_cond=True,
    drop_text_rate=0.1,
    unconditional_guidance_scale=5,
    use_SR=False,
    merging_SR=False,
    decoder_lambda_entropy=0.,
    bg_color=1,
    rc_loss=dict(
        type='MSELoss',
        loss_weight=20.0),
    cache_size=501989,
    EMpose=False,
    num_file_writers=8,
    autocast_dtype=None,
    cache_16bit=True)

save_interval = 10000
eval_interval = 10000
code_dir = 'cache/' + name + '/code'
load_code_dir = 'nerfs/' + name + '/code'
work_dir = 'work_dirs/' + name

train_cfg = dict()
test_cfg = dict(
    img_size=(256, 256),
    num_timesteps=50,
    clip_range=[-2, 2],
    density_thresh=0.1,
    save_dir=work_dir + '/saved_mesh',
    save_mesh=True,
    mesh_resolution=256,
    mesh_threshold=10,
)

dataset_type = 'Direct3D_demo'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    val_uncond=dict(
        type=dataset_type,
        data_prefix='data/test',
        load_imgs=False,
        TPose=False,
        RotAngle=False,
        num_test_imgs=50, # The number of images you want to render, can be a value from 1-251. Depends on your GPU memory. (Rendering 251 images takes lots of memeory..)
        random_test_imgs=True,
        scene_id_as_name=True,
        max_num_scenes=4,
        cache_path='data/test_cache.pkl'),
    train_dataloader=dict(split_data=True, check_batch_disjoint=True))

checkpoint_config = dict(interval=save_interval, by_epoch=False, max_keep_ckpts=100)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_uncond',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=32,
        viz_dir=work_dir + '/viz_uncond',
        save_best_ckpt=False)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
