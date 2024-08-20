#!/usr/bin/env python

from setuptools import setup

setup(
    name="edm_tts",
    version="0.0.1",
    description="EDM-TTS implementation",
    author="Anonymous Author",
    packages=["edm_tts"],
    package_data={
        "edm_tts": ["models/semantic_tokenizer_hubert/semantic_cluster_centers/*.pt"]
    },
    include_package_data=True,
)
