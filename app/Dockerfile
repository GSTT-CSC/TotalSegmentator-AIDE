# Use initial MAP for build
FROM ghcr.io/gstt-csc/totalsegmentator-aide/map-init:0.1.0 AS build

# Add dcm2niix to MAP
WORKDIR /bin

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
	git build-essential cmake pigz

RUN git clone https://github.com/rordenlab/dcm2niix.git --branch master --single-branch \
	&& cd dcm2niix \
	&& mkdir build && cd build \
	&& cmake .. \
	&& make

ENV PATH="$PATH:/bin/dcm2niix/build/bin"

# Add TotalSegmentator and rt-utils to MAP
WORKDIR /home

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    libsm6 libxrender-dev libxext6 ffmpeg

RUN python -m pip install TotalSegmentator
RUN python -m pip install rt-utils

# Workaround for opencv package issue
# see here: https://stackoverflow.com/questions/72706073/attributeerror-partially-initialized-module-cv2-has-no-attribute-gapi-wip-gs
RUN python -m pip uninstall -y opencv-python
RUN python -m pip install opencv-python==4.5.5.64

WORKDIR /var/monai
