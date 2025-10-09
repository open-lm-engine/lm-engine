# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

install:
	git submodule update --init --recursive
	make -C flash-model-architectures install
	uv env
	uv pip install .

install-dev:
	git submodule update --init --recursive
	make -C flash-model-architectures install
	uv env
	uv pip install -e .

test:
	RUN_SLOW=True uv run pytest tests

test-fast:
	RUN_SLOW=False uv run pytest tests

update-precommit:
	uv run pre-commit autoupdate

style:
	uv run python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run pre-commit run --all-files
