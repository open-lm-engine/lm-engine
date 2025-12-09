FILE=data

docker build -f docker/$FILE.Dockerfile --platform linux/amd64 -t ghcr.io/open-lm-engine/$FILE:latest .
