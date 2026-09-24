"""
tinygpt.py - a bare-bones GPT you train from scratch on ANY text file.

The model code never changes. Swap the text file to change what it learns.

Usage:
  python tinygpt.py train data.txt              # train on data.txt, saves tinygpt.pt
  python tinygpt.py generate "Tracer says"      # generate text from a prompt

Needs: Python 3.10+, PyTorch with CUDA (https://pytorch.org/get-started/locally/)
"""
import sys, time, torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- settings (tuned for an 8GB GPU like a 3070 Ti) ----------------
CKPT        = "tinygpt.pt"
batch_size  = 64      # how many text chunks per training step
block_size  = 256     # how many characters of context the model can "see"
n_embd      = 384     # size of each token's vector
n_head      = 6       # attention heads per layer
n_layer     = 6       # number of transformer blocks (~10M parameters total)
dropout     = 0.2     # randomly drops connections during training to reduce memorizing
lr          = 3e-4    # learning rate: how big each weight nudge is
max_iters   = 5000    # training steps (~15-30 min on a 3070 Ti)
eval_every  = 250     # how often to print progress
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


# ---------------- the model ----------------
class SelfAttention(nn.Module):
    """Lets each character look back at earlier characters to gather context."""
    def __init__(self):
        super().__init__()
        self.qkv  = nn.Linear(n_embd, 3 * n_embd)   # query, key, value in one go
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        # split into heads: (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
        # is_causal=True: a token can only look at the past, never the future
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    """One transformer block: attention (communicate) + MLP (think)."""
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = SelfAttention()
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # "+ x" = residual connection, keeps training stable
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # character ID -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)   # position -> vector
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size)      # vector -> score for every possible next char

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            # how wrong were the next-character guesses?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Predict one character, append it, repeat."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ---------------- training ----------------
def train(data_file):
    text = open(data_file, encoding="utf-8").read()
    if len(text) < 20 * block_size:
        sys.exit(f"{data_file} is only {len(text):,} characters. Aim for 100k+ (1M+ is better).")

    # the tokenizer: every unique character gets an ID number
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    split = int(0.9 * len(data))              # 90% to learn from, 10% to test on
    train_data, val_data = data[:split], data[split:]

    def get_batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # targets = shifted by one
        return x.to(device), y.to(device)

    model = GPT(len(chars)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} | {len(text):,} chars | vocab={len(chars)} | params={n_params/1e6:.1f}M")

    use_amp = device == "cuda"
    best_val = float("inf")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for name, d in (("train", train_data), ("val", val_data)):
            losses = []
            for _ in range(50):
                x, y = get_batch(d)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, loss = model(x, y)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    t0 = time.time()
    for step in range(max_iters + 1):
        if step % eval_every == 0:
            l = estimate_loss()
            print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} "
                  f"| {time.time() - t0:.0f}s")
            if l["val"] < best_val:   # only keep the best version
                best_val = l["val"]
                torch.save({"model": model.state_dict(), "chars": chars}, CKPT)
            # peek at what it's learned so far
            sample = model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 120)
            print("   sample:", repr("".join(chars[i] for i in sample[0].tolist())))

        x, y = get_batch(train_data)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()      # figure out which direction to nudge every weight
        opt.step()           # nudge them

    print(f"done. best val loss {best_val:.3f}, saved to {CKPT}")


# ---------------- generating ----------------
def generate(prompt, length=500):
    ckpt = torch.load(CKPT, map_location=device)
    chars = ckpt["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    model = GPT(len(chars)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ids = [stoi[c] for c in prompt if c in stoi] or [0]   # drop chars it never saw
    out = model.generate(torch.tensor([ids], device=device), length)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        train(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "generate":
        generate(sys.argv[2] if len(sys.argv) > 2 else "\n")
    else:
        print(__doc__)
