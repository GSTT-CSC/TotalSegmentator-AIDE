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

WORKDIR /var/monai


# Add TotalSegmentator to MAP

# TODO: add TotalSegmentator
