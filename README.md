# makemore_benchmark

> Bigram language model benchmark — **count-based vs. neural**, implemented following Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) lecture series.

Measures both **sampling speed** and **generalization accuracy (NLL)** on a 90/10 train/test split of 32K English first names.

---

## Report

[View full benchmark report](https://htmlpreview.github.io/?https://github.com/seungjae99/makemore_benchmark/blob/main/benchmark_results.html)

---

## Results

### Accuracy — Test NLL Loss (lower is better)

| Model | Test NLL |
|---|---|
| Count-based bigram | **2.4511** (best) |
| Neural bigram | 2.4712 |

### Speed — Sampling Latency (lower is better)

| Model | ms / sample | Relative Speed |
|---|---|---|
| Count-based bigram | **0.0489** | 1.00× (baseline) |
| Neural (no\_grad) | 0.0979 | 2.00× slower |
| Neural (with grad) | 0.1165 | 2.38× slower |

> Mean over 1,000 samples · fixed seed `42`

**Count-based wins on both dimensions** — it is ~2× faster and achieves lower NLL.

---

## Models

### Count-based Bigram
Builds a 27×27 character bigram frequency matrix over the train split, applies Laplace smoothing (+1), and normalises each row into a probability distribution `P`. Sampling is a single row lookup followed by `torch.multinomial`.

### Neural Bigram
Trains a single 27×27 weight matrix `W` with gradient descent on the train split (100 epochs, lr = 50, L2 λ = 0.01). Sampling runs a full forward pass — one-hot encoding → matmul → exp → normalise — at every character step.

---

## Key Findings

**Count-based is faster AND more accurate at bigram scale.**
For a 27-character vocabulary, a smoothed frequency table is the optimal approach — it has no arithmetic overhead at sample time and Laplace smoothing is a provably well-calibrated prior for discrete bigram data.

**`torch.no_grad()` saves only ~19% of neural inference time.**
The bottleneck is the forward arithmetic (matmul + exp + normalise), not autograd bookkeeping.

**Neural becomes worthwhile when the architecture scales.**
Once the context window grows beyond 1 character (MLP / Transformer blocks), a lookup table is no longer tractable — that is where the neural path earns its overhead.

---

## Usage

```bash
python3 benchmark.py
```

```
Dataset: 32033 words  →  train 28829 / test 3204

Test NLL loss  (22,567 bigrams from 3,204 held-out words)
  count-based : 2.4511
  neural      : 2.4712

model                             ms/sample   relative speed
------------------------------------------------------------
count-based bigram                  0.0489            1.00x
neural (with grad)                  0.1165            2.38x
neural (no_grad)                    0.0979            2.00x
```

## Dataset

`names.txt` is not included in this repository. Download it from Andrej Karpathy's makemore:

```bash
curl -O https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```

## File Structure

```
makemore_benchmark/
├── benchmark.py        # entry point — trains, evaluates loss, measures speed
├── count_bigram.py     # count-based model (build + sample)
├── neural_bigram.py    # neural model (train + sample)
└── names.txt           # download separately (see Dataset section above)
```

## Environment

| | |
|---|---|
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| Platform | macOS 26.5 (arm64) |
| Train / Test split | 90 / 10 (seed 42) |
| Samples per speed run | 1,000 |
