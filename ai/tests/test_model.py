import unittest
import jax
import jax.numpy as jnp
from src.model import LanguageModel

class TestModel(unittest.TestCase):
    def setUp(self):
        def model_fn(tokens):
            return LanguageModel(vocab_size=100, model_size=16, num_layers=1, num_q_heads=2, num_kv_heads=2, key_size=8, widening_factor=4.0, num_experts=1, use_quant=False)(tokens)
        self.model = hk.transform(model_fn)
        self.rng = jax.random.PRNGKey(42)
        self.params = self.model.init(self.rng, jnp.zeros((1, 10), dtype=jnp.int32))
    
    def test_forward(self):
        output = self.model.apply(self.params, None, jnp.zeros((1, 10), dtype=jnp.int32))
        self.assertEqual(output.shape, (1, 10, 100))

if __name__ == "__main__":
    unittest.main()