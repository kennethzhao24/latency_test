#!/bin/bash

B=1
L=1024
MODEL="cerebras-gpt"
# MODEL="gpt2"
# MODEL="opt-350m"
# MODEL="opt-125m"
# MODEL="opt-100m"
# MODEL="opt-80m"
# MODEL="opt-60m"
# MODEL="pythia-70m"
# MODEL="pythia-160m"





python inference_test.py \
   --model_name $MODEL \
   --batch_size $B \
   --seq_len $L \
   --cuda
