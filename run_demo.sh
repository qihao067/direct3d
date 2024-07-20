CFG=text_to_3d
CKPT=checkpoint/path #TODO: put the checkpoint path here

rm -rf cache/
rm -rf work_dirs/

python3 test.py ./configs/${CFG}.py ${CKPT}  --gpu-ids 0 --inference_prompt 'a brown boot' --seed 99