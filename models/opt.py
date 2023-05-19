""" 
    OPT Transformer Model.
"""
from typing import Optional


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from .modules import OPTLearnedPositionalEmbedding, OPTLayer
from .utils import _expand_mask, _make_causal_mask



class OPTBlock(nn.Module):
    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 head_dim: int,
                 ffn_dim: int,
                 attention_dropout: float = 0.0,
                 dropout: float = 0.0,
                 activation_function: str = "relu"):
        super().__init__()

        self.layers = nn.ModuleList([OPTLayer(hidden_size,
                                              head_dim,
                                              ffn_dim,
                                              attention_dropout,
                                              dropout,
                                              activation_function) for _ in range(num_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False):

        all_hidden_states = ()
        all_self_attns = ()

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            outputs = layer(hidden_states,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            use_cache=use_cache)

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attns += (outputs[1],)
        
        return hidden_states, all_hidden_states, all_self_attns


class OPTBlockWS(nn.Module):
    """
        Transformer Block w/ parameter sharing
    """
    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 head_dim: int,
                 ffn_dim: int,
                 attention_dropout: float = 0.0,
                 dropout: float = 0.0,
                 activation_function: str = "relu"):
        super().__init__()

        self.num_layers = num_layers
        self.layer = OPTLayer(hidden_size, head_dim, ffn_dim, attention_dropout, dropout, activation_function)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False):

        all_hidden_states = ()
        all_self_attns = ()

        for _ in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            outputs = self.layer(hidden_states,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            use_cache=use_cache)

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attns += (outputs[1],)
        
        return hidden_states, all_hidden_states, all_self_attns


class OPTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size[0])

        # project in layer if neeeded
        if config.word_embed_proj_dim != config.hidden_size[0]:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size[0], bias=False)
        else:
            self.project_in = None

        # project out layer if neeeded
        if config.word_embed_proj_dim != config.hidden_size[-1]:
            self.project_out = nn.Linear(config.hidden_size[-1], config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        self.final_layer_norm = nn.LayerNorm(config.hidden_size[-1])

        if config.weight_sharing:
            self.blocks = nn.ModuleList([OPTBlockWS(config.layers_per_block[i],
                                                config.hidden_size[i],
                                                config.head_dim,
                                                config.ffn_dim[i],
                                                config.attention_dropout,
                                                config.dropout,
                                                config.activation_function)
                                                for i in range(config.num_blocks)])        
        else:
            self.blocks = nn.ModuleList([OPTBlock(config.layers_per_block[i],
                                                config.hidden_size[i],
                                                config.head_dim,
                                                config.ffn_dim[i],
                                                config.attention_dropout,
                                                config.dropout,
                                                config.activation_function)
                                                for i in range(config.num_blocks)])
            
        self.block_proj = nn.ModuleList([nn.Identity() for i in range(config.num_blocks)])

        for i in range(config.num_blocks-1):   
            if config.hidden_size[i] != config.hidden_size[i+1]:
                self.block_proj[i] = nn.Linear(config.hidden_size[i], config.hidden_size[i+1], bias=False)
        

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # retrieve input_ids
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        else:
            raise ValueError("You have to specify either decoder_input_ids")

        past_key_values_length = 0

        # generate input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # generate position embeddings
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)        
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        # prepare decoder attention masks
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        # prepare hidden states
        hidden_states = inputs_embeds + pos_embeds

        # print(hidden_states.shape)
        # print(attention_mask.shape)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = block_outputs[0]

            if output_hidden_states:
                all_hidden_states += (block_outputs[1],)

            if output_attentions:
                all_self_attns += (block_outputs[2],)

            # print(1)
            hidden_states =  self.block_proj[idx](hidden_states)  
            # print(2)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)


class OPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = OPTDecoder(config)
        # Initialize weights
        self._init_weights(self.decoder)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None):

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return decoder_outputs


class OPTForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = OPTModel(config)

        self.config = config
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        # enable weight sharing between lm_head weight and embed tokens weight
        self.lm_head.weight = self.model.decoder.embed_tokens.weight

        # Initialize weights
        self._init_weights(self.model)
        self._init_weights(self.lm_head)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`OPTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.

        Returns:

        Example:

        ```python
        >>> from transformers import GPT2Tokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return tuple(v for v in [loss, logits] if v is not None)

