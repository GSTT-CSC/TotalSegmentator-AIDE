# Use initial MAP for build
FROM ghcr.io/gstt-csc/totalsegmentator-aide/map-init:0.1.2 AS build

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

ENV WEIGHTS_DIR="/root/.totalsegmentator/nnunet/results/nnUNet/3d_fullres/"
RUN mkdir -p ${WEIGHTS_DIR}

# Part 1 - Organs
ENV WEIGHTS_URL_1="https://zenodo.org/record/6802342/files/Task251_TotalSegmentator_part1_organs_1139subj.zip"
ENV WEIGHTS_ZIP_1="Task251_TotalSegmentator_part1_organs_1139subj.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL_1} \
    && unzip ${WEIGHTS_DIR}${WEIGHTS_ZIP_1} -d ${WEIGHTS_DIR} \
    && rm ${WEIGHTS_DIR}${WEIGHTS_ZIP_1}

# Part 2 - Vertebrae
ENV WEIGHTS_URL_2="https://zenodo.org/record/6802358/files/Task252_TotalSegmentator_part2_vertebrae_1139subj.zip"
ENV WEIGHTS_ZIP_2="Task252_TotalSegmentator_part2_vertebrae_1139subj.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL_2} \
    && unzip ${WEIGHTS_DIR}${WEIGHTS_ZIP_2} -d ${WEIGHTS_DIR} \
    && rm ${WEIGHTS_DIR}${WEIGHTS_ZIP_2}

# Part 3 - Cardiac
ENV WEIGHTS_URL_3="https://zenodo.org/record/6802360/files/Task253_TotalSegmentator_part3_cardiac_1139subj.zip"
ENV WEIGHTS_ZIP_3="Task253_TotalSegmentator_part3_cardiac_1139subj.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL_3} \
    && unzip ${WEIGHTS_DIR}${WEIGHTS_ZIP_3} -d ${WEIGHTS_DIR} \
    && rm ${WEIGHTS_DIR}${WEIGHTS_ZIP_3}

# Part 4 - Muscles
ENV WEIGHTS_URL_4="https://zenodo.org/record/6802366/files/Task254_TotalSegmentator_part4_muscles_1139subj.zip"
ENV WEIGHTS_ZIP_4="Task254_TotalSegmentator_part4_muscles_1139subj.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL_4} \
    && unzip ${WEIGHTS_DIR}${WEIGHTS_ZIP_4} -d ${WEIGHTS_DIR} \
    && rm ${WEIGHTS_DIR}${WEIGHTS_ZIP_4}

# Part 5 - Ribs
ENV WEIGHTS_URL_5="https://zenodo.org/record/6802452/files/Task255_TotalSegmentator_part5_ribs_1139subj.zip"
ENV WEIGHTS_ZIP_5="Task255_TotalSegmentator_part5_ribs_1139subj.zip"

RUN wget --directory-prefix ${WEIGHTS_DIR} ${WEIGHTS_URL_5} \
    && unzip ${WEIGHTS_DIR}${WEIGHTS_ZIP_5} -d ${WEIGHTS_DIR} \
    && rm ${WEIGHTS_DIR}${WEIGHTS_ZIP_5}

# Set TOTALSEG_WEIGHTS_PATH ENV variable – this is auto-detected by TotalSegmentator
# See: https://github.com/wasserth/TotalSegmentator/blob/f4651171a4c6eae686dd67b77efe6aa78911734d/totalsegmentator/libs.py#L77
ENV TOTALSEG_WEIGHTS_PATH="/root/.totalsegmentator/nnunet/results/"

WORKDIR /var/monai
