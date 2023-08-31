#!/bin/bash


export PYTHONPATH=$PYTHNPATH:/home/fsuser/vllm-quantisation

/usr/local/NVIDIA-Nsight-Compute/ncu \
	-o iterate.ncu-rep \
	--target-processes all \
	--set full \
	-f \
       	-c 5 \
	--import-source on \
	-s 20 \
	-k "quant_forward_mm" \
	--source-folders /home/fsuser/vllm-quantisation/new_ext/new_kernels \
       	python3 test.py


# -k "regex:single_query_cached_kv_attention.*" \
	#--nvtx \
	#--call-stack \
