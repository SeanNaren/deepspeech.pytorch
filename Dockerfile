FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

WORKDIR /workspace/

# install basics 
RUN apt-get update -y
RUN apt-get install -y git curl ca-certificates bzip2 cmake tree htop bmon iotop

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
RUN /miniconda/bin/conda install conda-build \
 && /miniconda/bin/conda create -y --name py36 python=3.6.4 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Ensure conda version is at least 4.4.11
# (because of this issue: https://github.com/conda/conda/issues/6811)
ENV CONDA_AUTO_UPDATE_CONDA=false
RUN conda install -y "conda>=4.4.11" && conda clean -ya

RUN conda install pytorch torchvision cuda91 -c pytorch && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y \
    h5py \
 && conda clean -ya
RUN pip install h5py-cache

# install python deps
RUN pip install cython cffi tensorboardX

# install warp-CTC

ENV CUDA_HOME=/usr/local/cuda
RUN apt-get install -y g++
RUN git clone https://github.com/SeanNaren/warp-ctc.git
RUN cd warp-ctc; mkdir build; cd build; cmake ..; make
RUN cd warp-ctc; cd pytorch_binding; python setup.py install

# install pytorch audio
RUN apt-get install -y sox libsox-dev libsox-fmt-all
RUN git clone https://github.com/pytorch/audio.git
RUN cd audio; python setup.py install

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip install .

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# install deepspeech2 pytorch implementation
ADD . /workspace/deepspeech.pytorch
RUN cd deepspeech.pytorch; pip install -r requirements.txt

# launch jupiter
RUN pip install jupyter
RUN mkdir data; mkdir notebooks;
CMD jupyter-notebook --ip="*" --no-browser --allow-root
