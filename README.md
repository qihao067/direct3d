# DIRECT-3D (CVPR2024)
This is the official PyTorch implementation of the paper:

[CVPR24'] DIRECT-3D: Learning Direct Text-to-3D Generation on Massive Noisy 3D Data

[Qihao Liu](https://qihao067.github.io/) | [Yi Zhang](https://edz-o.github.io/) | [Song Bai](https://songbai.site/) | [Adam Kortylewski](https://gvrl.mpi-inf.mpg.de/) | [Alan Yuille](https://cogsci.jhu.edu/directory/alan-yuille/) 

[[project page]()] | [[paper]()] | [[arxiv]()]

______

## TODO

- [ ] Release all pretrained checkpoints (before June 14th)
- [ ] Release code to improve DreamFusion (before June 23th)

______

## Requirements

The code has been tested with PyTorch 2.1.0 and Cuda 12.1.

A example of installation commands is provided as follows:

```
#### Install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

#### Install MMCV and MMGeneration
pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip3 install -e .

#### Install other dependencies
pip install -r requirements.txt

#### Install other cuda related package
cd lib/ops/raymarching/
pip install -e .
cd ../shencoder/
pip install -e .
cd ../../..
```



______

## Checkpoints

We will release all checkpoints very soon (before June 14th). 



______

## Text to 3D generation

Run the following command to generate 3D objects. Both 3D meshs and 2D rendered images will be saved.

```
python3 test.py ./configs/text_to_3d.py /path/to/checkpoint --gpu-ids 0 --inference_prompt 'a dinosaur' --seed 99
```

You may also run  `run_demo.sh` to for text-to-3D generation

```
bash run_demo.sh
```



______

## Improving 2D-lifting Methods with 3D Prior

We will release the code very soon. 

______

## Acknowledgements

This codebase is built upon the following repositories:

- SSDNeRF
- Stable Diffusion

____________

## License

The code in this repository is released under the MIT License

______

## BibTeX

```
TODO
```

