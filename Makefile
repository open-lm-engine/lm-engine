# **************************************************
# Copyright (c) 2026, Mayank Mishra, Aditya Prasad, Shawn Tan
# **************************************************

accelerator=cuda

docker-data:
	docker build -f docker/Dockerfile --platform linux/amd64 --build-arg EXTRA=data -t ghcr.io/open-lm-engine/data:latest .

docker-tpu:
	docker build -f docker/Dockerfile --platform linux/amd64 --build-arg EXTRA=tpu -t ghcr.io/open-lm-engine/tpu:latest .

test:
	RUN_SLOW=True uv run --extra dev --extra torch pytest tests

test-fast:
	RUN_SLOW=False uv run --extra $(accelerator) --extra dev pytest tests

update-precommit:
	uv run --extra dev --no-default-groups pre-commit autoupdate

style:
	uv run --extra dev --no-default-groups python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) $$(date +%Y), __authors__"
	uv run --extra dev --no-default-groups pre-commit run --all-files
