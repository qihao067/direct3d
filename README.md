# DIRECT-3D (CVPR2024)

This is the official PyTorch implementation of the paper:

[CVPR24'] DIRECT-3D: Learning Direct Text-to-3D Generation on Massive Noisy 3D Data

[Qihao Liu](https://qihao067.github.io/) | [Yi Zhang](https://edz-o.github.io/) | [Song Bai](https://songbai.site/) | [Adam Kortylewski](https://gvrl.mpi-inf.mpg.de/) | [Alan Yuille](https://cogsci.jhu.edu/directory/alan-yuille/) 

[[project page](https://direct-3d.github.io/)] | [[paper](https://arxiv.org/pdf/2406.04322)] | [[arxiv](https://arxiv.org/abs/2406.04322)]

______

**DIRECT-3D is a new text-to-3D generative model that directly generates 3D contents in a single forward pass without optimization.**

- **[Fast Text-to-3D generation without optimization]** It can generate high-quality 3D objects with accurate geometric details and various textures in 12 seconds on a single V100, driven by text prompts.

  ![teaser2](https://github.com/qihao067/direct3d/blob/main/imgs/teaser2.gif)

- **[Accurate 3D geometry prior]** It also provides accurate and effective 3D geometry prior that significantly alleviates the Janus problem in 2D-lifting methods. The 3D knowledge is embedded in a stable diffusion-like architecture, ensuring ease of use and compatibility with many existing algorithms.

  ![teaser3](https://github.com/qihao067/direct3d/blob/main/imgs/teaser3.gif)

______

## TODO

- [ ] Release all pretrained checkpoints (before the end of June)
- [ ] Release code to improve DreamFusion (before the end of June)

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
cd ..

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

We will release all checkpoints very soon (before the end of June). 



______

## Text to 3D generation

Run the following command to generate 3D objects. Both 3D meshes and 2D rendered images will be saved.

```
python3 test.py ./configs/text_to_3d.py /path/to/checkpoint --gpu-ids 0 --inference_prompt 'a dinosaur' --seed 99
```

You may also run  `run_demo.sh` for text-to-3D generation.

```
bash run_demo.sh
```



______

## Improving 2D-lifting Methods with 3D Prior

We will release the code very soon. 

______

## Acknowledgements

This codebase is built upon the following repositories:

- [[SSDNeRF](https://github.com/Lakonik/SSDNeRF)]
- [[Stable Diffusion](https://github.com/CompVis/stable-diffusion)]

Much appreciation for the outstanding efforts.

____________

## License

The code in this repository is released under the MIT License

______

## BibTeX

```
@inproceedings{liu2024direct,
  title={DIRECT-3D: Learning Direct Text-to-3D Generation on Massive Noisy 3D Data},
  author={Liu, Qihao and Zhang, Yi and Bai, Song and Kortylewski, Adam and Yuille, Alan L},
  booktitle={41st IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024},
  organization={IEEE}
}
```

