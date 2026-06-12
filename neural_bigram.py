import torch
import torch.nn.functional as F

def build_neural_model(words):
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    
    g = torch.Generator().manual_seed(990901)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    
    xenc = F.one_hot(xs, num_classes=27).float()
    for _ in range(100):
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
        W.grad = None
        loss.backward()
        W.data += -50 * W.grad # type: ignore
    
    return W.detach(), stoi, itos

def sample_neural(W, itos, g):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    return ''.join(out[:-1])