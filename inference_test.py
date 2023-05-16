import argparse
import json
import torch

from deepspeed.profiling.flops_profiler import get_model_profile
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM, AutoModelForCausalLM

from models import OPTModel
from utils import measure_inference_latency





def parse_args():
    parser = argparse.ArgumentParser(description="Inference Test")
    parser.add_argument("--model_name", type=str, help="model name", 
                        choices=["EleutherAI/pythia-160m-deduped", 
                                 "EleutherAI/pythia-70m-deduped",
                                 "facebook/opt-125m",
                                 "facebook/opt-350m",
                                 "cerebras/Cerebras-GPT-111M",
                                 "gpt2",
                                 "opt"], 
                        default='opt')
    parser.add_argument("--config_file", type=str, help="Config file", default='.')
    parser.add_argument("--ws", action="store_true", help="Use weight sharing or not")
    parser.add_argument("--cuda", action="store_true", help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, help="Batch size.", default=1)
    parser.add_argument("--seq_len", type=int, help="Sequence length.", default=1024)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if 'pythia' in args.model_name:
        model = GPTNeoXForCausalLM.from_pretrained(
                args.model_name,
                revision="step3000",
                )
    elif 'gpt' in args.model_name:
        config = AutoConfig.from_pretrained('gpt2')
        model = GPT2LMHeadModel(config)
    elif args.model_name == 'opt':
        with open(args.config_file) as f:
            data = json.loads(f.read())
        from types import SimpleNamespace
        config = SimpleNamespace(**data)
        config.weight_sharing = False
        if args.ws:
            config.weight_sharing = True
        print(config)
        model = OPTModel(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            )

    def input_constructor(batch_size, seq_len):
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        fake_seq = ""
        for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
            fake_seq += tokenizer.pad_token
        inputs = tokenizer([fake_seq] * batch_size,
                        padding=True,
                        truncation=True,
                        return_tensors="pt")
        inputs = dict(inputs)
        return inputs

    with torch.cuda.device(0):
        flops, macs, params = get_model_profile(model,
                    kwargs=input_constructor(args.batch_size, args.seq_len),
                    print_profile=False,
                    detailed=False
                    )
        
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    
    latency = measure_inference_latency(model, 
                                        n_threads=1, 
                                        seq_len=args.seq_len, 
                                        n_trials=16,
                                        device=device) * 1000

    print("Param size = {}".format(params))
    print("FLOPs = {}".format(flops))
    print("MACs = {}".format(macs))
    print("Latency = {:.2f} ms".format(latency))


if __name__ == "__main__":
    main()