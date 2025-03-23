#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    required = f.read().splitlines()
    # 移除注释行和空行
    required = [line for line in required if line and not line.startswith('#')]

setup(
    name="smoke_detection",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="烟雾环境下的人体目标检测系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/smoke_detection",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=required,
    entry_points={
        "console_scripts": [
            "smoke_demo=smoke_detection.demo.example:main",
        ],
    },
) 