import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from .model import LanguageModel
from .processor import TextProcessor
from .data_augmentation import augment_data
import joblib

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data[idx]["input"]
        target_text = self.data[idx]["response"]
        return input_text, target_text

def collate_fn(batch, tokenizer):
    inputs = []
    targets = []
    for input_text, target_text in batch:
        input_tokens = tokenizer.encode(input_text, max_length=128, truncation=True)
        target_tokens = tokenizer.encode(target_text, max_length=128, truncation=True)
        if len(input_tokens) > 1 and target_tokens:
            inputs.append(input_tokens)
            targets.append(target_tokens[0])
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)

def train_model():
    tweets = load_dataset("tweets_tr", split="train")
    opus = load_dataset("opus_open_subtitles", "tr-en", split="train")
    augment_data('data/chat_data.txt', "data/augmented_chat_data.txt")
    with open("data/augmented_chat_data.txt", 'r', encoding='utf-8') as f:
        local_lines = f.read().splitlines()
    
    data = []
    for i in range(0, len(local_lines)-1, 2):
        data.append({"input": local_lines[i], "response": local_lines[i+1]})
    for tweet in tweets[:1000]:
        data.append({"input": tweet["text"], "response": "Hmmm, ilginç! 😊"})
    for dialog in opus[:1000]:
        data.append({"input": dialog["tr"], "response": dialog["en"]})
    
    tokenizer = TextProcessor().tokenizer
    dataset = ChatDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                           collate_fn=lambda batch: collate_fn(batch, tokenizer))

    def model_fn(tokens):
        model = LanguageModel(
            vocab_size=32000,
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

    model = hk.transform(model_fn)
    rng = jax.random.PRNGKey(42)
    params = model.init(rng, np.zeros((1, 128), dtype=np.int32))
    opt = jax.optim.adam(0.001)
    opt_state = opt.init(params)

    def loss_fn(params, tokens, targets):
        logits = model.apply(params, None, tokens)
        return jnp.mean(jax.nn.log_softmax(logits)[:, targets])

    @jax.jit
    def update(params, opt_state, tokens, targets):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
        updates, opt_state = opt.update(grads, opt_state)
        params = jax.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    for epoch in range(20):
        total_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            params, opt_state, loss = update(params, opt_state, inputs, targets)
            total_loss += loss
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dataloader)}")
        joblib.dump(params, "checkpoints/chatbot.pkl")