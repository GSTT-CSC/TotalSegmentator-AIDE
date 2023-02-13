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

# Add TotalSegmentator model weights to container
WORKDIR /root

ENV WEIGHTS_DIR_1="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj"
ENV WEIGHTS_URL_1="https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
ENV WEIGHTS_ZIP_1="Task256_TotalSegmentator_3mm_1139subj.zip"

RUN mkdir ${TOTALSEG_WEIGHTS_PATH} \
    && wget --directory-prefix ${TOTALSEG_WEIGHTS_PATH} ${WEIGHTS_URL} \
    && unzip ${WEIGHTS_DIR}${WEIGHTS_ZIP_1} -d ${TOTALSEG_WEIGHTS_PATH} \
    && rm ${TOTALSEG_WEIGHTS_PATH}${WEIGHTS_ZIP_1}

# Set TOTALSEG_WEIGHTS_PATH ENV variable – this is auto-detected by TotalSegmentator
# See: https://github.com/wasserth/TotalSegmentator/blob/f4651171a4c6eae686dd67b77efe6aa78911734d/totalsegmentator/libs.py#L77
ENV TOTALSEG_WEIGHTS_PATH="/root/.totalsegmentator/nnunet/results/"

WORKDIR /var/monai
