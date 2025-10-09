# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

install:
	pip install -r requirements.txt
	git submodule update --init --recursive
	cd flash-model-architectures && make install

install-dev:
	pip install -r requirements-dev.txt
	git submodule update --init --recursive
	cd flash-model-architectures && make install

test:
	RUN_SLOW=True uv run pytest tests

test-fast:
	RUN_SLOW=False pytest tests

update-precommit:
	pre-commit autoupdate

style:
	uv run python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run pre-commit run --all-files
