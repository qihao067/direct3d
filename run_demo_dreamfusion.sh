EXP="photo_corgi_wearing_crown_dressed_like_Henry_VIII_king_of_England_in_a_garden_getty_photo"
DIRECT3D_PROMPT="a corgi"
PROMPT="a zoomout photo of a corgi wearing a crown dressed like Henry VIII king of England, Getty Images"

WEIGHT=direct3d_small_0.002_copyema
SEED=$RANDOM
GPU=0
DIRECT3D_CFG=7
TWOD_CFG=50
PRIOR_WEIGHT=1e-1
NEGA_W=-3 
CFG_RESCALE=0.2
ENTROPY_WEIGHT=5e1
FOVY=33
LR=1e-4
NEGA_PROMPT="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

# SD
WORK_DIR="work_dirs_dreamfusion/direct3d_sd/${WEIGHT}_cfg${DIRECT3D_CFG}/${EXP}/cfg${TWOD_CFG}_prior${PRIOR_WEIGHT}_seed$SEED"

CUDA_VISIBLE_DEVICES=$GPU python demo_dreamfusion.py --config configs/text_to_3d_dreamfusion.py --checkpoint ckpts/${WEIGHT}.pth --gpu-ids 0 \
--seed $SEED --text "$PROMPT" --prior_text "${DIRECT3D_PROMPT}" --negative "${NEGA_PROMPT}" --workspace ${WORK_DIR} \
--geo_only -O \
--lr $LR --lambda_entropy ${ENTROPY_WEIGHT} \
--perpneg --negative_w ${NEGA_W} \
--guidance_scale ${TWOD_CFG} --default_fovy $FOVY \
--max_ray_batch 2048 \
--lambda_direct3d_prior 6000 ${PRIOR_WEIGHT} 0 9000 \
--coarse_to_fine --ws 64 128 --hs 64 128 --resolution_milestones 5000 \
--save_mesh --mcubes_resolution 128 \
--cfg-options test_cfg.num_timesteps=50 model.unconditional_guidance_scale=${DIRECT3D_CFG}

# Recon loss, IF
WORK_DIR="work_dirs_dreamfusion/direct3d_if/${WEIGHT}_cfg${DIRECT3D_CFG}/${EXP}/cfg${TWOD_CFG}_cfgscale${CFG_RESCALE}_prior${PRIOR_WEIGHT}_seed$SEED"

CUDA_VISIBLE_DEVICES=$GPU python demo_dreamfusion.py --config configs/text_to_3d_dreamfusion.py --checkpoint ckpts/${WEIGHT}.pth --gpu-ids 0 \
--seed $SEED --text "$PROMPT" --prior_text "${DIRECT3D_PROMPT}" --negative "${NEGA_PROMPT}" --workspace ${WORK_DIR} \
--IF --vram_O \
--geo_only -O \
--lr $LR --lambda_entropy ${ENTROPY_WEIGHT} \
--recon_loss --recon_std_rescale ${CFG_RESCALE} \
--perpneg --negative_w ${NEGA_W} \
--guidance_scale ${TWOD_CFG} --default_fovy $FOVY \
--max_ray_batch 2048 \
--lambda_direct3d_prior 6000 ${PRIOR_WEIGHT} 0 9000 \
--coarse_to_fine --ws 64 96 --hs 64 96 --resolution_milestones 5000 \
--save_mesh --mcubes_resolution 128 \
--cfg-options test_cfg.num_timesteps=50 model.unconditional_guidance_scale=${DIRECT3D_CFG}
