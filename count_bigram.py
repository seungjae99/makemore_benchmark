import torch

def build_count_model(words):
    N = torch.zeros((27, 27), dtype=torch.int32)
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    
    P = (N + 1).float()
    P /= P.sum(1, keepdim=True)
    
    return P, stoi, itos

def sample_count(P, itos, g):
    out = []
    ix = 0
    while True:
        ix = int(torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item())
        out.append(itos[ix])
        if ix == 0:
            break
    return ''.join(out[:-1])