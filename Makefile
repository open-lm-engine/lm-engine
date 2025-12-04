# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

accelerator=cuda

test:
	RUN_SLOW=True uv run --extra dev --extra flash-attn --extra xma pytest tests

test-fast:
	RUN_SLOW=False uv run --extra $(accelerator) --extra dev pytest tests

update-precommit:
	uv run --extra dev --no-default-groups pre-commit autoupdate

style:
	uv run --extra dev --no-default-groups python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run --extra dev --no-default-groups pre-commit run --all-files
