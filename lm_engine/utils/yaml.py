# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
import re

import yaml


def load_yaml(file_path: str) -> dict:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(file_path, "r") as f:
        content = f.read()

    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
    content = pattern.sub(lambda match: os.environ.get(match.group(1), match.group(0)), content)

    return yaml.load(content, loader)
