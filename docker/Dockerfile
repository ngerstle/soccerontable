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

ENV BASEDIR="/opt"
################################ Install the COCO API

RUN git clone https://github.com/cocodataset/cocoapi.git ${BASEDIR}/cocoapi
WORKDIR ${BASEDIR}/cocoapi/PythonAPI
RUN make -j`nproc` install

RUN pip2.7 install pycocotools && pip3 install pycocotools

############################### DETECTRON
# Clone the Detectron repository
ENV DETECTRON="${BASEDIR}/detectron"
RUN git clone https://github.com/facebookresearch/detectron ${DETECTRON}

# Go to Detectron root
WORKDIR ${BASEDIR}/detectron

# Install Python dependencies
RUN pip3 install -r ${BASEDIR}/detectron/requirements.txt
RUN pip2.7 install -r ${BASEDIR}/detectron/requirements.txt

# Set up Python modules & [Optional] Build custom ops
RUN make -j`nproc` && make -j`nproc` ops



###########################OPENCV

# opencv stuff


WORKDIR ${BASEDIR}

ENV OPENCV_VERSION="3.4.1" 

RUN wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& rm ${OPENCV_VERSION}.zip

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir ${BASEDIR}/opencv-${OPENCV_VERSION}/cmake_binary 


WORKDIR ${BASEDIR}/opencv-${OPENCV_VERSION}/cmake_binary 
RUN  cmake -DBUILD_TIFF=ON \
	  -DBUILD_opencv_java=OFF \
	  -DOPENCV_EXTRA_MODULES_PATH=${BASEDIR}/opencv_contrib-${OPENCV_VERSION}/modules \
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

ENV OPENPOSEDIR="${BASEDIR}/openpose"
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose ${OPENPOSEDIR}
WORKDIR ${OPENPOSEDIR}
RUN mkdir -p build && cd build \
	&& cmake \ 
		-DOpenCV_CONFIG_FILE=${BASEDIR}/opencv-${OPENCV_VERSION}/cmake_binary/OpenCVConfig.cmake \
		 .. \
	&& cmake -DCMAKE_BUILD_TYPE=Release .. \
	&& make -j"$(nproc)"

RUN cd models && ./getModels.sh

WORKDIR ${OPENPOSEDIR}
RUN ./build/openpose


###################### SOCCERONTABLE

WORKDIR ${BASEDIR}
ENV SOCCERCODE="${BASEDIR}/soccerontable"
ENV DATADIR="${BASEDIR}/datadir"
ENV MODELPATH="${BASEDIR}/model.pth"

ENV PYTHONPATH="${PYTHONPATH}:${SOCCERCODE}:${OPENPOSEDIR}:${DETECTRON}"

RUN git clone https://github.com/ngerstle/soccerontable.git ${SOCCERCODE}

WORKDIR ${SOCCERCODE}
RUN chmod u+x compile.sh \
 && ./compile.sh

WORKDIR ${SOCCERCODE}/utils/nms
RUN python3 setup.py build_ext --inplace
RUN python2 setup.py build_ext --inplace

WORKDIR ${SOCCERCODE}
RUN pip3 install -r requirements.txt
RUN pip2.7 install -r requirements.txt

## get example dataset
RUN mkdir -p ${DATADIR}
WORKDIR ${BASEDIR}
RUN wget http://grail.cs.washington.edu/projects/soccer/barcelona.zip \
 && unzip barcelona.zip \
 && mv barcelona/*  ${DATADIR}

RUN pip2.7 install pycocotools h5py
RUN pip3 install pycocotools h5py
RUN mkdir -p  ${DATADIR}/detectron

WORKDIR ${SOCCERCODE
RUN cp ${SOCCERCODE}/utils/thirdpartyscripts/infer_subimages.py ${DETECTRON}/tools/


COPY model.pth ${BASEDIR}/model.pth
COPY runsoccercode.sh ${BASEDIR}/runsoccercode.sh
RUN chmod u+x ${BASEDIR}/runsoccercode.sh


CMD ["/bin/bash"]
