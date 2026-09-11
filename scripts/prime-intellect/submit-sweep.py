# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import argparse
import itertools
import json
import os
import re
import subprocess
import time

import jinja2
import jinja2.meta


CONFIG_PATH = "configs/research/delta-mlp/config_rendered.yml"
SUBMIT_PATH = "scripts/prime-intellect/submit.sh"


def parse_fixed(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise ValueError(f"Fixed param must be in key=value format, got: {s!r}")
    k, v = s.split("=", maxsplit=1)
    k = k.strip()
    v = v.strip()
    return k, v


def parse_sweep(s: str) -> tuple[str, list[str]]:
    if "=" not in s:
        raise ValueError(f"Sweep param must be in key=v1,v2,... format, got: {s!r}")
    k, vs = s.split("=", maxsplit=1)
    k = k.strip()
    vs = [v.strip() for v in vs.split(",")]
    return k, vs


def _sanitize(text: str) -> str:
    text = text.lower()

    # Replace whitespace and separators with underscore
    text = re.sub(r"[\/\\:\s]+", "_", text)

    # Keep only safe characters
    text = re.sub(r"[^a-z0-9._-]", "", text)

    # Collapse repeats
    text = re.sub(r"_+", "_", text)

    # Trim junk from ends
    text = text.strip("._-")

    assert text
    return text


def dict_to_filename(params: dict[str, object]) -> str:
    assert params, f"params must not be empty, got: {params}"
    entries = []
    for key, val in params.items():
        sanitized_key = _sanitize(str(key))
        sanitized_val = _sanitize(str(val))
        entries.append(f"{sanitized_key}_{sanitized_val}")

    return "__".join(entries)


def render_template(template_path: str, params: dict) -> str:
    template_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)
    loader = jinja2.FileSystemLoader(template_dir)
    env = jinja2.Environment(loader=loader, undefined=jinja2.StrictUndefined)
    template = env.get_template(template_name)

    # this avoids having unused parameters
    template_source, _, _ = loader.get_source(env, template_name)
    template_ast = env.parse(template_source)
    template_vars = jinja2.meta.find_undeclared_variables(template_ast)

    unused = set(params.keys()) - template_vars
    if unused:
        raise ValueError(f"Unused template parameters: {sorted(unused)}")

    return template.render(**params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--fixed", type=str, nargs="+", default=None)
    parser.add_argument("--sweep", type=str, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.template):
        raise ValueError
    if not os.path.exists(SUBMIT_PATH):
        raise ValueError

    fixed_params = {}
    sweep_params = {}
    if args.fixed is not None:
        for fixed_arg in args.fixed:
            fixed_key, fixed_val = parse_fixed(fixed_arg)
            assert fixed_key not in fixed_params.keys()
            assert fixed_key not in sweep_params.keys()
            fixed_params[fixed_key] = fixed_val

    if args.sweep is not None:
        for sweep_arg in args.sweep:
            sweep_key, sweep_vals = parse_sweep(sweep_arg)
            assert sweep_key not in fixed_params.keys()
            assert sweep_key not in sweep_params.keys()
            assert len(sweep_vals) >= 1
            sweep_params[sweep_key] = sweep_vals

    sweep_keys = sweep_params.keys()
    for sweep_val in itertools.product(*sweep_params.values()):
        _sweep_params = dict(zip(sweep_keys, sweep_val))
        if _sweep_params:
            _name = dict_to_filename(_sweep_params)
            name = f"{args.name}/{_name}"
        else:
            name = args.name

        params = {
            "group": args.name,
            "notes": name,
            **fixed_params,
            **_sweep_params,
        }
        config = render_template(args.template, params)
        with open(CONFIG_PATH, "w") as f:
            f.write(config)

        diff_cmd = [
            "git",
            "--no-pager",
            "diff",
            "--no-index",
            args.template,
            CONFIG_PATH,
        ]
        submit_cmd = [
            "bash",
            SUBMIT_PATH,
            "--name",
            name,
            "--config",
            CONFIG_PATH,
            "--max-length",
            str(args.max_length),
        ]
        print(json.dumps(params, indent=2))
        print(f"\n[INFO] {' '.join(diff_cmd)}")
        subprocess.run(diff_cmd, check=False)
        print(f"\n[INFO] {' '.join(submit_cmd)}")
        if not args.dry_run:
            subprocess.run(submit_cmd, check=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
