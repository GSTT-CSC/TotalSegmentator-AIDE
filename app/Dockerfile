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

# Part 1 - Organs
ENV WEIGHTS_DIR_1="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj"
ENV WEIGHTS_URL_1="https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip?download=1"
ENV WEIGHTS_ZIP_1="Task251_TotalSegmentator_part1_organs_1139subj.zip"

RUN mkdir -p ${WEIGHTS_DIR_1} \
    && wget --directory-prefix ${WEIGHTS_DIR_1} ${WEIGHTS_URL_1} \
    && unzip ${WEIGHTS_DIR_1}${WEIGHTS_ZIP_1} -d ${WEIGHTS_DIR_1} \
    && rm ${WEIGHTS_DIR_1}${WEIGHTS_ZIP_1}

# Part 2 - Vertebrae
ENV WEIGHTS_DIR_2="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj"
ENV WEIGHTS_URL_2="https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip?download=1"
ENV WEIGHTS_ZIP_2="Task252_TotalSegmentator_part2_vertebrae_1139subj.zip"

RUN mkdir -p ${WEIGHTS_DIR_2} \
    && wget --directory-prefix ${WEIGHTS_DIR_2} ${WEIGHTS_URL_2} \
    && unzip ${WEIGHTS_DIR_2}${WEIGHTS_ZIP_2} -d ${WEIGHTS_DIR_2} \
    && rm ${WEIGHTS_DIR_2}${WEIGHTS_ZIP_2}

# Part 3 - Cardiac
ENV WEIGHTS_DIR_3="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj"
ENV WEIGHTS_URL_3="https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip?download=1"
ENV WEIGHTS_ZIP_3="Task253_TotalSegmentator_part3_cardiac_1139subj.zip"

RUN mkdir -p ${WEIGHTS_DIR_3} \
    && wget --directory-prefix ${WEIGHTS_DIR_3} ${WEIGHTS_URL_3} \
    && unzip ${WEIGHTS_DIR_3}${WEIGHTS_ZIP_3} -d ${WEIGHTS_DIR_3} \
    && rm ${WEIGHTS_DIR_3}${WEIGHTS_ZIP_3}

# Part 4 - Muscles
ENV WEIGHTS_DIR_4="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj"
ENV WEIGHTS_URL_4="https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip?download=1"
ENV WEIGHTS_ZIP_4="Task254_TotalSegmentator_part4_muscles_1139subj.zip"

RUN mkdir -p ${WEIGHTS_DIR_4} \
    && wget --directory-prefix ${WEIGHTS_DIR_4} ${WEIGHTS_URL_4} \
    && unzip ${WEIGHTS_DIR_4}${WEIGHTS_ZIP_4} -d ${WEIGHTS_DIR_4} \
    && rm ${WEIGHTS_DIR_4}${WEIGHTS_ZIP_4}

# Part 5 - Ribs
ENV WEIGHTS_DIR_5="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/Task251_TotalSegmentator_part1_organs_1139subj"
ENV WEIGHTS_URL_5="https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip?download=1"
ENV WEIGHTS_ZIP_5="Task255_TotalSegmentator_part5_ribs_1139subj.zip"

RUN mkdir -p ${WEIGHTS_DIR_5} \
    && wget --directory-prefix ${WEIGHTS_DIR_5} ${WEIGHTS_URL_5} \
    && unzip ${WEIGHTS_DIR_5}${WEIGHTS_ZIP_5} -d ${WEIGHTS_DIR_5} \
    && rm ${WEIGHTS_DIR_5}${WEIGHTS_ZIP_5}

# Set TOTALSEG_WEIGHTS_PATH ENV variable – this is auto-detected by TotalSegmentator
# See: https://github.com/wasserth/TotalSegmentator/blob/f4651171a4c6eae686dd67b77efe6aa78911734d/totalsegmentator/libs.py#L77
ENV TOTALSEG_WEIGHTS_PATH="/root/.totalsegmentator/nnunet/results/"

WORKDIR /var/monai
