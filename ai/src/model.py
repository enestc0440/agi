import jax
import jax.numpy as jnp
import haiku as hk
from jax.sharding import PartitionSpec as P
from typing import Any, Optional, Union, Tuple, List
from .triton_kernels import act_quant, weight_dequant

class QuantizedWeight8bit:
    def __init__(self, weight: jnp.array, scales: jnp.array):
        self.weight = weight
        self.scales = scales

    @property
    def shape(self):
        return self.weight.shape

jax.tree_util.register_pytree_node(
    QuantizedWeight8bit,
    lambda qw: ([qw.weight, qw.scales], ()),
    lambda _, children: QuantizedWeight8bit(children[0], children[1]),
)

def with_sharding_constraint(x, constraint):
    return x  # JAX sharding yerel ortamda devre dışı

def cast_bfloat16(x):
    if x.dtype.kind == "f":
        return x.astype(jnp.bfloat16)
    return x

def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8
    return _ffn_size

class KVMemory:
    def __init__(self, k: Optional[jax.Array], v: Optional[jax.Array], step: Optional[jax.Array]):
        self.k = k
        self.v = v
        self.step = step

class Memory:
    def __init__(self, layers: List[KVMemory]):
        self.layers = layers

class Router(hk.Module):
    def __init__(self, num_selected_experts: int, name: str = "router"):
        super().__init__(name)
        self.num_selected_experts = num_selected_experts

    def compute_routing_prob(self, inputs: jax.Array, num_experts: int):
        inputs = jax.lax.convert_element_type(inputs, jnp.float32)
        routing_logits = self._router_weights(inputs, num_experts)
        routing_probs = jax.nn.softmax(routing_logits)
        return routing_probs, routing_logits, 0

    def _router_weights(self, x: jax.Array, num_experts: int):
        input_size = x.shape[-1]
        w = hk.get_parameter("w", [input_size, num_experts], jnp.float32, init=hk.initializers.Constant(0))
        return jnp.dot(x, w)

class MoELayer(hk.Module):
    def __init__(self, num_experts: int, layer_fn, router: Router, name: str = "moe"):
        super().__init__(name)
        self.num_experts = num_experts
        self.layer_fn = layer_fn
        self.router = router

    def __call__(self, inputs: jax.Array):
        routing_probs, _, _ = self.router.compute_routing_prob(inputs, self.num_experts)
        expert_gate, expert_index = jax.lax.top_k(routing_probs, k=self.router.num_selected_experts)
        tmp = jnp.reshape(inputs, (inputs.shape[0] * inputs.shape[1], inputs.shape[2]))
        broad_inputs = jnp.tile(tmp[:, jnp.newaxis, :], (1, self.router.num_selected_experts, 1))
        broad_inputs = jnp.reshape(broad_inputs, (broad_inputs.shape[0] * broad_inputs.shape[1], broad_inputs.shape[2]))
        init_fn, _ = hk.transform(self.layer_fn)
        vmapped_init_fn = jax.vmap(init_fn, in_axes=0, out_axes=0)
        lifted_init_fn = hk.experimental.transparent_lift(vmapped_init_fn)
        params = lifted_init_fn(jax.random.split(jax.random.PRNGKey(1), self.num_experts), jnp.zeros((self.num_experts, 1, 1, inputs.shape[-1])))
        return inputs  # Basitleştirilmiş, tam MoE sonra eklenebilir

class MultiHeadAttention(hk.Module):
    def __init__(self, num_q_heads: int, num_kv_heads: int, key_size: int, model_size: int, name: str = "mha"):
        super().__init__(name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.model_size = model_size

    def __call__(self, query: jax.Array, key: jax.Array, value: jax.Array):
        query_heads = self._linear_projection(query, self.key_size, self.num_q_heads, name="query")
        key_heads = self._linear_projection(key, self.key_size, self.num_kv_heads, name="key")
        value_heads = self._linear_projection(value, self.key_size, self.num_kv_heads, name="value")
        attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads)
        attn_weights = jax.nn.softmax(attn_logits)
        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
        final_projection = hk.Linear(self.model_size, name="linear")
        return final_projection(jnp.reshape(attn, (*attn.shape[:2], -1)))

    def _linear_projection(self, x: jax.Array, head_size: int, num_heads: int, name: str):
        y = hk.Linear(num_heads * head_size, name=name)(x)
        leading_dims, _ = x.shape
        return y.reshape((leading_dims, num_heads, head_size))

class DenseBlock(hk.Module):
    def __init__(self, model_size: int, widening_factor: float, name: str = "dense"):
        super().__init__(name)
        self.model_size = model_size
        self.widening_factor = widening_factor

    def __call__(self, inputs: jax.Array):
        ffn_size_val = ffn_size(self.model_size, self.widening_factor)
        h_v = hk.Linear(ffn_size_val, name="linear_v")(inputs)
        h_w1 = jax.nn.gelu(hk.Linear(ffn_size_val, name="linear")(inputs))
        return hk.Linear(self.model_size, name="linear_1")(h_w1 * h_v)

class DecoderLayer(hk.Module):
    def __init__(self, num_q_heads: int, num_kv_heads: int, key_size: int, widening_factor: float, num_experts: int, name: str = "decoder"):
        super().__init__(name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.widening_factor = widening_factor
        self.num_experts = num_experts

    def __call__(self, inputs: jax.Array):
        h = hk_rms_norm(inputs)
        attn = MultiHeadAttention(self.num_q_heads, self.num_kv_heads, self.key_size, inputs.shape[-1])(h, h, h)
        h = h + hk_rms_norm(attn)
        if self.num_experts > 1:
            router = Router(num_selected_experts=1)
            dense = MoELayer(self.num_experts, lambda x: DenseBlock(inputs.shape[-1], self.widening_factor)(x), router)
        else:
            dense = DenseBlock(inputs.shape[-1], self.widening_factor)
        h_dense = dense(hk_rms_norm(h))
        return h + hk_rms_norm(h_dense)

class Transformer(hk.Module):
    def __init__(self, num_q_heads: int, num_kv_heads: int, key_size: int, num_layers: int, widening_factor: float, num_experts: int, name: str = "transformer"):
        super().__init__(name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.num_experts = num_experts

    def __call__(self, embeddings: jax.Array):
        h = embeddings
        for i in range(self.num_layers):
            h = DecoderLayer(self.num_q_heads, self.num_kv_heads, self.key_size, self.widening_factor, self.num_experts, name=f"decoder_layer_{i}")(h)
        return h

class LanguageModel(hk.Module):
    def __init__(self, vocab_size: int, model_size: int, num_layers: int, num_q_heads: int, num_kv_heads: int, key_size: int, widening_factor: float, num_experts: int, use_quant: bool, name: str = "language_model"):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.widening_factor = widening_factor
        self.num_experts = num_experts
        self.use_quant = use_quant

    def __call__(self, tokens: jax.Array):
        in_out_embed = hk.Embed(vocab_size=self.vocab_size, embed_dim=self.model_size, name="in_out_embed")
        embeddings = in_out_embed(tokens).astype(jnp.bfloat16)
        transformer = Transformer(self.num_q_heads, self.num_kv_heads, self.key_size, self.num_layers, self.widening_factor, self.num_experts)
        h = transformer(embeddings)
        h = hk_rms_norm(h)
        logits = jnp.dot(h, in_out_embed.embeddings.T.astype(h.dtype))
        return logits

def hk_rms_norm(x: jax.Array):
    mean_squared = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(mean_squared + 1e-5)
    return normed_inputs