FROM pytorch

# based on https://github.com/SeanNaren/deepspeech.pytorch installation guidelines
# and of course work by from SeanNaren - kudos to him!
# 
# build command: 
#    $ nvidia-docker build -t  deepspeech2.docker .
# run command: 
#    $ nvidia-docker run -ti -v `pwd`/data:/workspace/data -p 8888:8888 deepspeech2.docker

# install basics 
RUN apt-get update -y
#RUN apt-get upgrade -y
RUN apt-get install -y git cmake tree htop bmon iotop

# install python deps
RUN pip install cffi tensorboardX

# install warp-CTC
RUN git clone https://github.com/SeanNaren/warp-ctc.git
RUN cd warp-ctc; mkdir build; cd build; cmake ..; make
RUN export CUDA_HOME="/usr/local/cuda"
RUN cd warp-ctc; cd pytorch_binding; python setup.py install

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
