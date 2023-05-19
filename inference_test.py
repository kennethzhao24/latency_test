import argparse
import json
from types import SimpleNamespace

from models import *
from utils import measure_inference_latency

model_to_config = {
    "pythia-160m": './configs/config_pythia_160m.json',
    "pythia-70m": './configs/config_pythia_70m.json',
    "opt-125m": './configs/config_opt_125m.json',
    "opt-350m": './configs/config_opt_350m.json',
    "cerebras-gpt": './configs/config_cerebras.json',
    "gpt2": './configs/config_gpt2.json',
    "opt-60m": './configs/config_params_60_20_ms.json',
    "opt-80m": './configs/config_params_80_29_ms.json',
    "opt-100m": './configs/config_params_100_43_ms.json'
}

def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Test")
    parser.add_argument("--model_name", type=str, help="model name", 
                        choices=["pythia-160m", 
                                 "pythia-70m",
                                 "opt-125m",
                                 "opt-350m",
                                 "cerebras-gpt",
                                 "gpt2",
                                 "opt-60m",
                                 "opt-80m",
                                 "opt-100m"
                                 ], 
                        default='opt-100m')
    parser.add_argument("--ws", action="store_true", help="Use weight sharing or not")
    parser.add_argument("--cuda", action="store_true", help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, help="Batch size.", default=1)
    parser.add_argument("--seq_len", type=int, help="Sequence length.", default=1024)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = model_to_config[args.model_name]
    with open(config_file) as f:
        data = json.loads(f.read())
    config = SimpleNamespace(**data)

    if 'pythia' in args.model_name:
        model = GPTNeoXForCausalLM(config)
    elif 'gpt' in args.model_name:
        model = GPT2LMHeadModel(config)
    elif 'opt' in args.model_name:
        config.weight_sharing = False
        if args.ws:
            config.weight_sharing = True
        model = OPTModel(config)
    else:
        raise ValueError('Model not Supported')
        
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    num_paranms = count_parameters(model)
    
    latency = measure_inference_latency(model, 
                                        n_threads=1, 
                                        seq_len=args.seq_len, 
                                        n_trials=16,
                                        device=device) * 1000
    
    print("Param size = {:.0f} m".format(num_paranms // 10 ** 6))
    print("Latency = {:.2f} ms".format(latency))


if __name__ == "__main__":
    main()