import torch
import torch.nn.functional as F
import time
from count_bigram import build_count_model, sample_count
from neural_bigram import build_neural_model, sample_neural

with open('names.txt', 'r') as f:
    words = f.read().splitlines()

# 90/10 train/test split
n = len(words)
torch.manual_seed(42)
idx = torch.randperm(n).tolist()
train_words = [words[i] for i in idx[:int(0.9 * n)]]
test_words  = [words[i] for i in idx[int(0.9 * n):]]

print(f"Dataset: {n} words  →  train {len(train_words)} / test {len(test_words)}\n")
print("Training models...")
P, stoi_c, itos_c = build_count_model(train_words)
W, stoi_n, itos_n = build_neural_model(train_words)
W_grad = W.clone().requires_grad_(True)
print("Done\n")

# --- Loss on test set ---
def to_bigram_tensors(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)

xs_test, ys_test = to_bigram_tensors(test_words, stoi_c)

count_loss = -P[xs_test, ys_test].log().mean().item()

with torch.no_grad():
    xenc = F.one_hot(xs_test, num_classes=27).float()
    logits = xenc @ W
    probs = logits.exp()
    probs /= probs.sum(1, keepdim=True)
    neural_loss = -probs[torch.arange(len(ys_test)), ys_test].log().mean().item()

print(f"Test NLL loss  ({len(xs_test):,} bigrams from {len(test_words):,} held-out words)")
print(f"  count-based : {count_loss:.4f}")
print(f"  neural      : {neural_loss:.4f}\n")

# --- Sampling speed ---
N_SAMPLES = 1000

g = torch.Generator().manual_seed(42)
start = time.perf_counter()
for _ in range(N_SAMPLES):
    sample_count(P, itos_c, g)
count_time = (time.perf_counter() - start) / N_SAMPLES * 1000

g = torch.Generator().manual_seed(42)
start = time.perf_counter()
for _ in range(N_SAMPLES):
    sample_neural(W_grad, itos_n, g)
neural_time = (time.perf_counter() - start) / N_SAMPLES * 1000

g = torch.Generator().manual_seed(42)
start = time.perf_counter()
with torch.no_grad():
    for _ in range(N_SAMPLES):
        sample_neural(W, itos_n, g)
neural_nograd_time = (time.perf_counter() - start) / N_SAMPLES * 1000

print(f"{'model':<30} {'ms/sample':>12} {'relative speed':>16}")
print("-" * 60)
print(f"{'count-based bigram':<30} {count_time:>11.4f} {'1.00x':>16}")
print(f"{'neural (with grad)':<30} {neural_time:>11.4f} {neural_time/count_time:>15.2f}x")
print(f"{'neural (no_grad)':<30} {neural_nograd_time:>11.4f} {neural_nograd_time/count_time:>15.2f}x")
