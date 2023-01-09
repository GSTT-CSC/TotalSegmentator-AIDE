# cd ~/TotalSegmentator-AIDE
# ensure venv running
# ensure Docker running

# Test MAP code locally
python app -i app/input -o app/output

# Test MAP with MONAI Deploy
monai-deploy exec app -i app/input -o app/output

# Initial packaging of MAP
monai-deploy package app --tag ghcr.io/gstt-csc/totalsegmentator-aide/map-init:0.1.0 -l DEBUG

# Push to GHCR
# - requires GH PAT
# - export export CR_PAT=<PAT>
echo $CR_PAT | docker login ghcr.io -u tomaroberts --password-stdin
docker push ghcr.io/gstt-csc/totalsegmentator-aide/map-init:0.1.0

# Build 3rd-party software on top of MAP
# CPU mode
git checkout map
docker build -t ghcr.io/gstt-csc/totalsegmentator-aide/map:0.1.0 app/

# Test MAP-Extra with MONAI Deploy
monai-deploy run ghcr.io/gstt-csc/totalsegmentator-aide/map:0.1.0 app/input app/output

# Optional: Test scripts within Docker container
# - On DGX:
docker run -it --rm -v /home/troberts/code/TotalSegmentator-AIDE/input:/home/recon --entrypoint /bin/bash ghcr.io/gstt-csc/totalsegmentator-aide/map:0.1.0