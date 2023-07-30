#!/bin/bash

export NCCL_IGNORE_DISABLED_P2P=1
#python3 -u -m vllm.entrypoints.api_server --model TheBloke/Wizard-Vicuna-13B-Uncensored-HF --tensor-parallel-size 2 --dtype half --swap-space 4
#python3 -u -m vllm.entrypoints.api_server --model ehartford/Wizard-Vicuna-13B-Uncensored --tensor-parallel-size 1 --dtype half --swap-space 4

pkill -f -9 ray
pkill -f -9 conda
sleep 30

python3 -u -m vllm.entrypoints.api_server --model TheBloke/Wizard-Vicuna-13B-Uncensored-HF --tensor-parallel-size 1 --dtype half --swap-space 4 --quantisation gptq
