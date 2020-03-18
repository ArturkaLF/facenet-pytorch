FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

MAINTAINER ArthurSh <sh.artur2000@gmail.com>

# make env python 3.6
RUN conda create -n py36 python=3.6 anaconda
# RUN echo "source activate myenv" > ~/.bashrc
# RUN conda activate py36

# install nvidia apex for amp
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /workspace/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./