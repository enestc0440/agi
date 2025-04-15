import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
from .model import LanguageModel
from .checkpoint import restore

class SampleSettings:
    def __init__(self, temperature, nucleus_p, mask, active):
        self.temperature = temperature
        self.nucleus_p = nucleus_p
        self.mask = mask
        self.active = active

class SampleOutput:
    def __init__(self, token_id, prob, top_k_token_ids, top_k_probs):
        self.token_id = token_id
        self.prob = prob
        self.top_k_token_ids = top_k_token_ids
        self.top_k_probs = top_k_probs

class InferenceRunner:
    def __init__(self, name, runner, load, tokenizer_path):
        self.name = name
        self.load = load
        self.tokenizer_path = tokenizer_path
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path)
        self.vocab_size = 32000
        self.max_len = 128
        self.batch_size = 1
        self.params = restore(load, None)

    def initialize(self):
        pass

    def run_prompt(self, prompt, max_len, temperature):
        tokens = np.array(self.tokenizer.encode(prompt), dtype=np.int32)
        tokens = np.pad(tokens, [0, self.max_len - len(tokens)], mode="constant", constant_values=0)
        
        def model_fn(tokens):
            model = LanguageModel(
                vocab_size=self.vocab_size,
                model_size=256,
                num_layers=2,
                num_q_heads=4,
                num_kv_heads=4,
                key_size=64,
                widening_factor=4.0,
                num_experts=1,
                use_quant=True
            )
            return model(tokens)
        
        model = hk.without_apply_rng(hk.transform(model_fn))
        logits = model.apply(self.params, None, tokens[np.newaxis, :])
        token_id = jax.random.categorical(jax.random.PRNGKey(42), logits[:, -1])
        return self.tokenizer.decode([int(token_id)])