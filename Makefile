# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

install:
	pip install -r requirements.txt
	git submodule update --init --recursive
	cd cute-kernels && make install

install-dev:
	pip install -r requirements-dev.txt
	git submodule update --init --recursive
	cd cute-kernels && make install

test:
	RUN_SLOW=True pytest tests

test-fast:
	RUN_SLOW=False pytest tests

update-precommit:
	pre-commit autoupdate

style:
	python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	pre-commit run --all-files
