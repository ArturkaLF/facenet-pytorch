ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"

WORKDIR /tmp/unique_for_apex

RUN pip uninstall -y apex || :

RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace

RUN pip install facenet-pytorch
