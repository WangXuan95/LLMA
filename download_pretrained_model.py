#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import huggingface_hub

# os.environ["http_proxy"]  = "http://xxxxxxx:xxxx"         # if need proxy, set it here
# os.environ["https_proxy"] = "http://xxxxxxx:xxxx"         # if need proxy, set it here

huggingface_token = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" # Your token generated in huggingface.co

huggingface_repo_id =                "Qwen/Qwen2.5-0.5B"    # model path in huggingface.co : https://huggingface.co/Qwen/Qwen2.5-0.5B
local_model_path    = "./model_params/Qwen/Qwen2.5-0.5B"    # the local path to save the model
cache_dir           =                         "./cache/"

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if not os.path.exists(local_model_path):
    os.makedirs(local_model_path)

huggingface_hub.snapshot_download(
    repo_id                = huggingface_repo_id,
    cache_dir              = cache_dir,
    local_dir              = local_model_path, 
    local_dir_use_symlinks = False,               # save as file rather than blob
    token                  = huggingface_token,
    force_download         = True,
    resume_download        = True
)
