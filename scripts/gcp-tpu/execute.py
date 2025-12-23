# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os
from argparse import ArgumentParser, Namespace

import yaml


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--script", required=True)

    args = parser.parse_args()

    return args


def main() -> None:
    args = get_args()
    with open(args.script, "r") as f:
        script = yaml.safe_load(f)

    if script["cloud"] == "gcp":
        project = script["node"]["project"]
        zone = script["node"]["zone"]
        name = script["node"]["name"]

        if script["spin-up"]:
            command = f"""gcloud compute tpus tpu-vm create {name} \
--project={project} \
--zone={zone} \
--version={script['node']['version']} \
--accelerator-type={script['node']['accelerator']}"""

            if script["node"].get("spot", False):
                command += " --spot"

            if script["node"].get("disks", None):
                for disk in script["node"]["disks"]:
                    command += (
                        f" --data-disk source=projects/{project}/zones/{zone}/disks/{disk['name']},mode=read-write"
                    )

            print(command)
            os.system(command)

        if script.get("installation", None):
            installation = "; ".join(script["installation"])

            command = f"""gcloud compute tpus tpu-vm ssh {name} \
--project={project} \
--zone={zone} \
--worker=all \
--command='{installation}'"""

            print(command)
            os.system(command)
    else:
        raise ValueError(f"Unsupported cloud provider: {script['cloud']}. Only 'gcp' is supported.")


if __name__ == "__main__":
    main()
