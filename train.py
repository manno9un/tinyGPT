import torch
import torch.nn as nn
from torch.nn import functional as F
import regex as re
import unicodedata
from datasets import load_dataset
import time
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

batch_size = 128
block_size = 512
max_iters = 15000
eval_interval = 250
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.15
# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================

def clean_text_english_only(text):
    text = re.sub(r'\p{Han}+', '[CJK]', text)
    text = re.sub(r'\p{Hiragana}+', '[JP]', text)
    text = re.sub(r'\p{Katakana}+', '[JP]', text)
    text = re.sub(r'\p{Hangul}+', '[KR]', text)
    text = re.sub(r'\p{Thai}+', '[THAI]', text)
    text = re.sub(r'\p{Arabic}+', '[AR]', text)
    text = re.sub(r'\p{Cyrillic}+', '[CYR]', text)
    text = re.sub(r'\p{Greek}+', '[GR]', text)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(char for char in text if char.isascii() or char == '\n')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text

if not os.path.exists('input.txt'):
    print("Loading and cleaning dataset...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    cleaned_ds = ds.map(lambda x: {"text": clean_text_english_only(x["text"])})
    with open("input.txt", "w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            for row in cleaned_ds[split]:
                f.write(row["text"] + "\n")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa, self.ffwd = MultiHeadAttention(n_head, head_size), FeedForward()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T)) if targets is not None else None
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            idx_next = torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    return torch.stack([d[i:i+block_size] for i in ix]).to(device), torch.stack([d[i+1:i+1+block_size] for i in ix]).to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.tensor([model(X, Y)[1].item() for X, Y in [get_batch(split) for _ in range(eval_iters)]])
        out[split] = losses.mean()
    model.train()
    return out

# ============================================================================
# RESUME LOGIC
# ============================================================================

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler();

if os.path.exists('model_weights.pt'):
    print("Found 'model_weights.pt'. Loading weights to resume training...")
    model.load_state_dict(torch.load('model_weights.pt', map_location=device))
    print("Weights loaded successfully.")
else:
    print("No existing weights found. Starting training from scratch.")

# ============================================================================
# TRAINING LOOP
# ============================================================================

start_time = time.time()
best_loss = float('inf')

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_loss:
            best_loss = losses['val']
            torch.save(model.state_dict(), 'model_weights.pt')
            print("weights saved val_loss:",best_loss)

    xb, yb = get_batch('train')

    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Save mappings for inference
with open('char_mappings.json', 'w') as f:
    json.dump({'stoi': stoi, 'itos': {str(k): v for k, v in itos.items()}, 'vocab_size': vocab_size}, f)
print("Training finished and artifacts updated.")