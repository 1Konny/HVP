FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y rsync htop git openssh-server sudo vim tmux

RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

#Torch and dependencies:
RUN pip install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp35-cp35m-linux_x86_64.whl 
RUN pip install https://download.pytorch.org/whl/cu90/torchvision-0.3.0-cp35-cp35m-manylinux1_x86_64.whl  
RUN pip install cffi tensorboardX
RUN pip install tqdm scipy scikit-image colorama==0.3.7 
RUN pip install setproctitle pytz ipython

#vid2vid dependencies
RUN apt-get install libglib2.0-0 libsm6 libxrender1 -y
RUN pip install dominate requests opencv-python 
