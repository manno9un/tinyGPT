"""
tGPT - Inference Script (Command-line)

This script loads a trained GPT model and generates text based on user input.

Usage:
    python app.py

The script loads:
    - model_weights.pt (trained model weights)
    - char_mappings.json (character mappings)

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FILE = "model_weights.pt"
MAPPINGS_FILE = "char_mappings.json"

print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}\n")

# ============================================================================
# CHECK FILES EXIST
# ============================================================================

if not os.path.exists(WEIGHTS_FILE):
    print(f"Error: {WEIGHTS_FILE} not found!")
    print(f"Please run: python train.py")
    sys.exit(1)

if not os.path.exists(MAPPINGS_FILE):
    print(f"Error: {MAPPINGS_FILE} not found!")
    print(f"Please run: python train.py")
    sys.exit(1)

# ============================================================================
# LOAD CHARACTER MAPPINGS
# ============================================================================

print("Loading character mappings...")
with open(MAPPINGS_FILE, 'r') as f:
    mappings = json.load(f)

stoi = mappings['stoi']
itos = {int(k): v for k, v in mappings['itos'].items()}
vocab_size = mappings['vocab_size']

print(f"Loaded {vocab_size} characters\n")

# Encoding and decoding
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# These must match the training script!
block_size = 512
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.0  # No dropout during inference



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
# LOAD TRAINED MODEL
# ============================================================================

print("Loading trained model...")
model = GPTLanguageModel().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=DEVICE))
model.eval()

print(" Model loaded successfully\n")

# ============================================================================
# INFERENCE LOOP
# ============================================================================

def generate_text(prompt, max_tokens=300):
    """
    Generate text from a prompt

    Args:
        prompt (str): Starting text
        max_tokens (int): Maximum tokens to generate
    """
    try:
        # Encode prompt
        encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

        # Generate
        with torch.no_grad():
            generated = model.generate(encoded, max_tokens)

        # Decode and return
        result = decode(generated[0].tolist())
        return result

    except KeyError as e:
      char = str(e).strip("'")
      return f"Error: Invalid character '{char}' in prompt"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main interactive loop"""
    print("="*70)
    print("tGPT - Text Generation Model")
    print("="*70)
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'help' - Show this help message")
    print("\nOptions:")
    print("  Type a prompt and press Enter to generate text")
    print("  You can control generation by modifying the settings below")
    print("="*70 + "\n")

    # Settings
    max_tokens = 300

    while True:
        print("\n" + "-"*70)
        print(f"Settings: max_tokens={max_tokens}")
        print("-"*70)

        user_input = input("Enter prompt (or command): ").strip()

        # Handle commands
        if not user_input:
            print(" Please enter a prompt or command")
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("\n Goodbye!")
            break

        if user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("  quit/exit - Exit")
            print("  help - Show this message")
            print("  tokens=N - Set max tokens (default 300)")
            print("  clear - Clear screen")
            continue

        if user_input.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            continue

        # Handle settings
        if user_input.lower().startswith('tokens='):
            try:
                max_tokens = int(user_input.split('=')[1])
                print(f"Max tokens set to {max_tokens}")
            except:
                print("Invalid format. Use: tokens=300")
            continue

        # Generate text
        print("\n Generating text...\n")
        result = generate_text(user_input, max_tokens)
        print("="*70)
        print(result)
        print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Interrupted by user. Goodbye!")
