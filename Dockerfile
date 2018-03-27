FROM floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.24

# install basics 
RUN apt-get update -y
RUN apt-get install -y git cmake tree htop bmon iotop

# install python deps
RUN pip install cffi tensorboardX

WORKDIR /workspace/

# install warp-CTC
RUN git clone https://github.com/SeanNaren/warp-ctc.git
RUN cd warp-ctc; mkdir build; cd build; cmake ..; make
RUN cd warp-ctc; cd pytorch_binding; CUDA_HOME="/usr/local/cuda" python setup.py install

# install pytorch audio
RUN apt-get install -y sox libsox-dev libsox-fmt-all
RUN git clone https://github.com/pytorch/audio.git
RUN cd audio; python setup.py install

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip install .

# install deepspeech2 pytorch implementation 
RUN git clone https://github.com/SeanNaren/deepspeech.pytorch
RUN cd deepspeech.pytorch; pip install -r requirements.txt

# launch jupiter
RUN pip install jupyter 
RUN mkdir data; mkdir notebooks;
CMD jupyter-notebook --ip="*" --no-browser --allow-root
