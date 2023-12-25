# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/llama/modeling_llama.py
from typing import List, Optional, Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel

from colossalai.inference.struct import BatchInfo


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def llama_causal_lm_forward(
    self: LlamaForCausalLM,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    hidden_states = llama_model_forward(
        self.model,
        batch=batch,
        k_caches=k_caches,
        v_caches=v_caches,
    )
    logits = self.lm_head(hidden_states)

    return logits


def llama_model_forward(
    self: LlamaModel,
    batch: BatchInfo = None,
    k_caches: List[torch.Tensor] = None,
    v_caches: List[torch.Tensor] = None,
):
    input_ids = batch.get_batch_inputs()
    block_tables = batch.get_block_table_tensor()
    sequence_lengths = batch.get_sequence_lengths()

    seq_length = input_ids.shape[1]
    device = input_ids.device

    past_key_values_length = len(block_tables.shape[1])

    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    hidden_states = self.embed_tokens(input_ids)

    for layer_id, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            position_ids=position_ids,
            block_tables=block_tables,
            k_cache=k_caches[layer_id],
            v_cache=v_caches[layer_id],
            is_prompts=batch.is_prompts,
            sequence_lengths=sequence_lengths,
        )

    hidden_states = self.norm(hidden_states)

    return hidden_states


def llama_decoder_layer_forward(
    self: LlamaDecoderLayer,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: int = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    hidden_states = self.self_attn(
        hidden_states=hidden_states,
        position_ids=position_ids,
        block_tables=block_tables,
        k_cache=k_cache,
        v_cache=v_cache,
        is_prompts=is_prompts,
        sequence_lengths=sequence_lengths,
    )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# Replace transformers.models.llama.modeling_llama.LlamaAttention.forward
def llama_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    block_tables: torch.Tensor = None,
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    is_prompts: bool = True,
    sequence_lengths: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2] + block_tables.shape[1]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    query_states = query_states.view(-1, self.num_heads, self.head_dim)
    key_states = key_states.view(-1, self.num_heads, self.head_dim)
    value_states = value_states.view(-1, self.num_heads, self.head_dim)

    block_size = k_cache.shape[-1]

    memcpy_to_block(key_states, value_states, k_cache, v_cache, block_tables, block_size)

    if is_prompts:
        attn_output = context_attention_unpadded(
            query_states, key_states, value_states, k_cache, v_cache, sequence_lengths, block_tables, block_size
        )
    else:
        attn_output = torch.empty(bsz, self.num_heads, self.head_dim)
        decoding_attention(
            query_states,
            k_cache,
            v_cache,
            block_tables,
            sequence_lengths,
            attn_output,
            block_tables.shape[1],
            block_size,
        )

    attn_output = attn_output.view(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output


def memcpy_to_block(key, value, k_cache, v_cache, block_tables, block_size):
    block_table_list = block_tables.tolist()
    batch_size, seq_len, num_heads, head_dim = key

    reshape_key = key.reshape(batch_size, seq_len, block_size, num_heads, head_dim).tensor.permute(0, 2, 3, 1)
    reshape_value = value.reshape(batch_size, seq_len, block_size, num_heads, head_dim).tensor.permute(0, 2, 3, 1)
    if seq_len == 1:
        for i in range(batch_size):
            k_cache[block_table_list[i][-1], :] = reshape_key[i]
            v_cache[block_table_list[i][-1], :] = reshape_value[i]
    else:
        for i in range(batch_size):
            k_cache[block_table_list[i], :] = reshape_key[i]
            v_cache[block_table_list[i], :] = reshape_value[i]
