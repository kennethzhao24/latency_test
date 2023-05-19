import json
from types import SimpleNamespace

config_file = './configs/config_pythia_160m.json'
with open(config_file) as f:
    data = json.loads(f.read())
config = SimpleNamespace(**data)

from models.gpt2 import GPT2LMHeadModel
from models.gptneox import GPTNeoXForCausalLM

# model = GPT2LMHeadModel(config)

model = GPTNeoXForCausalLM(config)

print(model)

from utils import measure_inference_latency

latency = measure_inference_latency(model, 
                                    n_threads=1, 
                                    seq_len=1024, 
                                    n_trials=16,
                                    device='cuda') * 1000
 
print("Latency = {:.2f} ms".format(latency))