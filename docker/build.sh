EXTRAS="data dev"
IMAGE=data

docker build -f docker/Dockerfile --platform linux/amd64 --build-arg EXTRAS=$EXTRAS -t ghcr.io/open-lm-engine/$IMAGE:latest .
