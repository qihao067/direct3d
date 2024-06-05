pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

#### Install MMCV and MMGeneration
pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmgeneration.git
cd mmgeneration
pip3 install -e .

#### install other dependencies
pip install -r requirements.txt

#### other cuda related package
cd lib/ops/raymarching/
pip install -e .
cd ../shencoder/
pip install -e .
cd ../../..