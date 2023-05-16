#!/bin/bash

B=1
L=128

python inference_test.py \
   --model_name EleutherAI/pythia-70m-deduped \
   --batch_size $B \
   --seq_len $L \
   --cuda


# python inference_test.py \
#    --model_name EleutherAI/pythia-160m-deduped \
#    --batch_size $B \
#    --seq_len $L \
#    --cuda


# python inference_test.py \
#    --model_name facebook/opt-125m \
#    --batch_size $B \
#    --seq_len $L \
#    --cuda

# python inference_test.py \
#    --model_name facebook/opt-350m \
#    --batch_size $B \
#    --seq_len $L \
#    --cuda

# python inference_test.py \
#    --model_name cerebras/Cerebras-GPT-111M \
#    --batch_size $B \
#    --seq_len $L \
#    --cuda


# python inference_test.py \
#    --model_name gpt2 \
#    --batch_size $B \
#    --seq_len $L \
#    --cuda


# python inference_test.py \
#    --model_name opt \
#    --config_file ./configs/config_params_60_20_ms.json \
#    --batch_size $B \
#    --seq_len $L \
#    --cuda \
#    --ws