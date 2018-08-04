# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda8.0-cudnn7-ubuntu16.04

RUN cp -r /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

########### UTILS

RUN apt-get update && \
	apt-get install -f -y \
		build-essential \
		cmake \
		curl \
		gcc \
		gfortran \
		git \
		libatlas-base-dev \
		libatlas-dev \
		libavcodec-dev \
		libavformat-dev \
		libboost-all-dev \
		libeigen3-dev \
		libgflags-dev \
		libgoogle-glog-dev \
		libgtk-3-dev  \
		libhdf5-dev \
		libjasper-dev \
		libjpeg-dev \
		libleveldb-dev \
		liblmdb-dev \
		libpng-dev \
		libpq-dev \
		libprotobuf-dev \
		libsnappy-dev \
		libswscale-dev \
		libtbb-dev \
		libtbb2 \
		libtiff-dev \
		libv4l-dev \
		libx264-dev \
		libxvidcore-dev \
		lsb-release \
		make \
		pkg-config \
		protobuf-compiler \
		python-dev \
		python-pip \ 
		python2.7 \
		python3-dev \ 
		python3-pip \
		python3.5 \
		python3.5-tk \
		qt5-default \
		screen \
		sudo \
		tmux \
		unzip \
		vim \
		vlc \
		wget \
		yasm 

RUN sed -i 's/geteuid/getppid/' /usr/bin/vlc

RUN pip2.7 install \
	cython \
	Cython \
	python-gflags \
	numpy \
	scipy \
	pyyaml \
	matplotlib \
	openmesh \
	glog \
	shapely \
	tqdm \
	torch \
	torchvision \
	visdom \
	pyopengl \
	setuptools

RUN pip3 install pipenv \
	cython \
	Cython \
	python-gflags \
	numpy \
	scipy \
	pyyaml \
	matplotlib \
	openmesh \
	glog \
	shapely \
	tqdm \
	torch \
	torchvision \
	visdom \
	pyopengl \
	setuptools

RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \
     &&	pip3 install torchvision

################################ Install the COCO API

RUN git clone https://github.com/cocodataset/cocoapi.git /opt/cocoapi
WORKDIR /opt/cocoapi/PythonAPI
RUN make -j`nproc` install

RUN pip2.7 install pycocotools && pip3 install pycocotools

############################### DETECTRON
# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/detectron /opt/detectron

# Go to Detectron root
WORKDIR /opt/detectron

# Install Python dependencies
RUN pip3 install -r /opt/detectron/requirements.txt
RUN pip2.7 install -r /opt/detectron/requirements.txt

# Set up Python modules & [Optional] Build custom ops
RUN make -j`nproc` && make -j`nproc` ops



###########################OPENCV

# opencv stuff


WORKDIR /opt

ENV OPENCV_VERSION="3.4.1" 

RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& rm ${OPENCV_VERSION}.zip

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opt/opencv-${OPENCV_VERSION}/cmake_binary 


WORKDIR /opt/opencv-${OPENCV_VERSION}/cmake_binary 
RUN  cmake -DBUILD_TIFF=ON \
	  -DBUILD_opencv_java=OFF \
	  -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
	  -DWITH_CUDA=OFF \
	  -DENABLE_AVX=ON \
	  -DWITH_OPENGL=ON \
	  -DWITH_OPENCL=ON \
	  -DWITH_IPP=ON \
	  -DWITH_TBB=ON \
	  -DWITH_EIGEN=ON \
	  -DWITH_V4L=ON \
	  -DWITH_GTK=OFF \
	  -DWITH_QT=ON \
	  -DBUILD_TESTS=OFF \
	  -DBUILD_PERF_TESTS=OFF \
	  -DINSTALL_PYTHON_EXAMPLES=ON \
	  -DCMAKE_BUILD_TYPE=RELEASE \
	  .. \
	&& make -j`nproc` install 

################################# OPENPOSE

RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose /opt/openpose
WORKDIR /opt/openpose
RUN mkdir -p build && cd build \
	&& cmake \ 
		-DOpenCV_CONFIG_FILE=/opt/opencv-${OPENCV_VERSION}/cmake_binary/OpenCVConfig.cmake \
		 .. \
	&& cmake -DCMAKE_BUILD_TYPE=Release .. \
	&& make -j"$(nproc)"

RUN cd models && ./getModels.sh

WORKDIR /opt/openpose
CMD ./build/openpose


######################

WORKDIR /opt


RUN sed -i 's/geteuid/getppid/' /usr/bin/vlc

COPY model.pth /opt/model.pth
COPY runsoccercode.sh /opt/runsoccercode.sh
RUN chmod u+x /opt/runsoccercode.sh

RUN ls -la /bin/bash

CMD ["/bin/bash"]
