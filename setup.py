# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from setuptools import find_packages, setup


with open("requirements.txt") as requirements:
    requirements = requirements.readlines()
    requirements = [i.strip() for i in requirements]

VERSION = "0.0.1.dev"

setup(
    name="lm-engine",
    version=VERSION,
    install_requires=requirements,
    author="Mayank Mishra",
    author_email="mayank31398@gmail.com",
    license="Apache License 2.0",
    url="https://github.com/open-lm-engine/lm-engine",
    packages=find_packages("./"),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
)
