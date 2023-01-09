# TotalSegmentator-AIDE

[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) packaged as an AIDE Application.

Currently... work in progress.

## Local Testing

1. Download
```shell
git clone https://github.com/GSTT-CSC/TotalSegmentator-AIDE.git
```

2. Setup virtual env
```shell
python -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Run

Make `input` and `output` directories. Copy DICOM file(s) into `input` directory.

```shell
monai-deploy exec app -i input/ -o -output/
```

## Build MAP

_TODO: Update once MAP has been created._