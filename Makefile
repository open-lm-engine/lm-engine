# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

test:
	RUN_SLOW=True pytest tests

test-fast:
	RUN_SLOW=False pytest tests

update-precommit:
	pre-commit autoupdate

style:
	python copyright/copyright.py --repo ./ --exclude copyright-exclude.txt --header "Copyright (c) 2025, Mayank Mishra"
	pre-commit run --all-files
