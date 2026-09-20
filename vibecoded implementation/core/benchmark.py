import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / d) for d in ("core", "graph", "demos", "tests")]

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

from main import Model

class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def __call__(self, x: mx.array): return self.proj(x)

def cola(filepath: str):
    data = []

    try:
        with open(filepath, 'r', encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4: data.append((parts[3].encode('utf-8'), int(parts[1])))

    except FileNotFoundError: pass
    
    return data

def mcc(tp, tn, fp, fn):
    import math
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    score = (tp * tn - fp * fn) / denominator if denominator != 0 else 0.0
    return score * 100

def run(path: str):
    model = Model(dim = 512, layers = 16, temp = 0.75, lr = 5e-4)
    model.load(path)
    model.freeze()

    head = Classification(model.dim)
    headopt = opt.AdamW(learning_rate = 1e-3)

    data = cola('CoLA/original/raw/in_domain_train.tsv')

    if data == []:
        print('invalid CoLA dataset.')
        return

    def lossfn(params, state: mx.array, target: int):
        head.update(params)
        choice = head(state)

        loss = nn.losses.cross_entropy(choice[None, :], mx.array([target])).mean()
        return loss, choice

    for epoch in range(3):
        print(f'\nEpoch {epoch + 1}')

        dummies = [mx.zeros((model.dim, )) for _ in range(model.layercount)]
        tp, tn, fp, fn = 0, 0, 0, 0
        
        for i, (b_s, label) in enumerate(data):
            model.reset()

            final = None
            for b in b_s:
                enc = model.encoder(mx.array(b))
                x = enc

                for j, layer in enumerate(model.layers):
                    x, state, _ = layer(enc, x, dummies[j])
                    layer.states = mx.stop_gradient(state)

                final = model.layers[-1].states

            (_, choice), grads = mx.value_and_grad(lossfn, argnums = 0)(head.trainable_parameters(), final, label)

            headopt.update(head, grads)
            mx.eval(head.parameters(), headopt.state)

            predicted_class = mx.argmax(choice).item()
            if predicted_class == 1 and label == 1: tp += 1
            elif predicted_class == 0 and label == 0: tn += 1
            elif predicted_class == 1 and label == 0: fp += 1
            elif predicted_class == 0 and label == 1: fn += 1

            score = mcc(tp, tn, fp, fn)

            if i > 0 and i % 500 == 0: print(f'{i}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({score:.4f})')

        print(f'{i}: T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({score:.4f})')

if __name__ == '__main__':
    # FIXME: 'experimental-4.5m.safetensors' is not shipped with this handoff (nor produced by any script here).
    # The checkpoints actually used are base-10m*.safetensors in checkpoints/ (dim=768, layers=16). Point this at one of those.
    run('experimental-4.5m.safetensors')