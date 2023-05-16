import torch
import torch.utils.benchmark as benchmark

from typing import Optional

def calcualte_param_analytical(config):
    params = {
        'embedding': 0,
        'attention': 0,
        'ff': 0,
        'layer_norm': 0,
        'non_embedding': 0,
        'total': 0
        }
    vocab_size = config.vocab_size
    max_position_embeddings = config.max_position_embeddings
    word_embed_proj_dim = config.word_embed_proj_dim
    hidden_size = config.hidden_size
    ffn_dim = config.ffn_dim
    num_blocks = config.num_blocks

    if config.weight_sharing:
        layers_per_block = [1] * 4
    else:
        layers_per_block = config.layers_per_block

    params['embedding'] = vocab_size * word_embed_proj_dim + (max_position_embeddings + 2) * hidden_size[0]

    # calculate project in if any
    if hidden_size[0] != word_embed_proj_dim:
        params['ff'] += word_embed_proj_dim * hidden_size[0]

    # caculate transformer params
    for i in range(num_blocks):
        params['attention'] += layers_per_block[i] *  (4 * hidden_size[i] * hidden_size[i] + 4 * hidden_size[i])
        params['ff'] += layers_per_block[i] * ((hidden_size[i] * ffn_dim[i] + ffn_dim[i]) + (ffn_dim[i] * hidden_size[i] + hidden_size[i]))
        params['layer_norm'] += layers_per_block[i] * 2 * hidden_size[i] * 2

    # calculate ffn params between blocks if any
    for i in range(num_blocks-1):   
        if hidden_size[i] != hidden_size[i+1]:
            params['ff'] += (hidden_size[i] * hidden_size[i+1])

    # calculate project out if any
    if hidden_size[-1] != word_embed_proj_dim:
        params['ff'] += word_embed_proj_dim * hidden_size[-1]    

    # caclulate final layer norm
    params['layer_norm'] += hidden_size[-1] * 2

    params['non_embedding'] = params['attention'] + params['ff'] + params['layer_norm']
    params['total'] = params['embedding'] + params['non_embedding']

    return params
 


def measure_inference_latency(model: torch.nn.Module,
                              use_median: Optional[bool] = False,
                              batch_size: Optional[int] = 1,
                              seq_len: Optional[int] = 256,
                              n_threads: Optional[int] = 1,
                              n_trials: Optional[int] = 10,
                              device: Optional[str] = 'cuda'):
    """Measures a model's inference latency.

    Args:
        model: Model instance.
        use_quantization: Whether latency should be calculated with quantizated model or not.
        use_median: Whether should use median instead of mean for latency measurement.
        batch_size: Batch size to measure the latency.
        seq_len: Sequence length to measure the latency.
        n_threads: Number of inference threads.
        n_trials: Number of times to repeat the measurement.
        device: Device where latency should be measured.

    Returns:
        (float): Mean or median latency in seconds.

    """
    torch.set_num_threads(n_threads)

    model = model.to(device=device)
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.int64).to(device)
    
    timer = benchmark.Timer(stmt='model(input_ids, labels, mems)',
                            globals={
                                'input_ids': input_ids,
                                'labels': None,
                                'mems': None,
                                'model': model
                            },
                            num_threads=n_threads)

    runner = timer.timeit(n_trials)

    return runner.median if use_median else runner.mean

