# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

test:
	RUN_SLOW=True uv run --extra dev --extra flash-attn pytest tests

test-fast:
	RUN_SLOW=False uv run --extra dev --extra flash-attn pytest tests

test-fast-cpu:
	RUN_SLOW=False uv run --extra dev pytest tests

update-precommit:
	uv run --extra dev pre-commit autoupdate

style:
	uv run --extra dev python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	uv run --extra dev pre-commit run --all-files
