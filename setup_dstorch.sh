pip install -r requirements.txt

#warp-ctc
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
cd ../..

#pytorch audio
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio && python setup.py install
cd ..

#NVIDIA apex
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
cd ..
