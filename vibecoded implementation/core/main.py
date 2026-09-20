import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils as util

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
        self.decay = mx.zeros((dim, ))
        self.states = mx.zeros((dim, ))

        self.decaytrace = mx.zeros((dim, ))
        self.embedtrace = mx.zeros((256, dim))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

    def __call__(self, enc: mx.array, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float, lr: float):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.layers = [Layer(dim) for _ in range(layers)]
        self.optimizer = opt.AdamW(learning_rate = lr)

    def sample(self, output: mx.array):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp)

    def reset(self):
        for layer in self.layers:
            layer.decay = mx.zeros((self.dim, ))
            layer.states = mx.zeros((self.dim, ))

            layer.decaytrace = mx.zeros((self.dim, ))
            layer.embedtrace = mx.zeros((256, self.dim))

    def step(self, c: mx.array, dummies: mx.array):
        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.layers):
            x, state, decay = layer(enc, x, dummies[i])

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def __call__(self, currb: int, nextb: int | None, end: bool, notrace: bool = False):
        c = mx.array(currb)

        if notrace:
            _, (output, stop) = self.step(c, [mx.zeros((self.dim, )) for _ in range(self.layercount)])
            return self.sample(output).item(), stop.item()

        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)
            (x, states, decays), (output, stop) = self.step(c, dummies)

            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4)) # variance
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                loss = loss + mx.mean(mx.square(x - tgt)) # pred mse
                loss = loss - output[n] + mx.logsumexp(output) # ce

                loss = loss + mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0]))) # stop mse

            return loss, (states, decays, output, stop) # loss = variance loss + pred mse loss + crossentropy loss + stop mse loss

        (_, (states, decays, output, stop)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layercount)])

        self.update(p)

        for i, layer in enumerate(self.layers):
            dlds = dlds_s[i]

            embedtrace = (layer.embedtrace * decays[i]) + (mx.arange(256) == c)[:, None]
            grads["encoder"]["embed"]["weight"] += dlds * (layer.embedtrace * decays[i])
            
            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)
            grads["layers"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])

            layer.decaytrace = mx.stop_gradient(decaytrace)
            layer.embedtrace = mx.stop_gradient(embedtrace)
            
            mx.eval(layer.states, layer.decaytrace, layer.embedtrace)

        self.optimizer.update(self, grads)
        mx.eval(self.parameters(), self.optimizer.state)

        return self.sample(output).item(), stop.item()

    def save(self, path: str):
        import os

        data = {}
        for k, v in util.tree_flatten(self.parameters()): data[f"m.{k}"] = v
        for k, v in util.tree_flatten(self.optimizer.state): data[f"o.{k}"] = v

        for i, layer in enumerate(self.layers):
            data[f"state.{i}"] = layer.states
            data[f"decaytrace.{i}"] = layer.decaytrace
            data[f"embedtrace.{i}"] = layer.embedtrace

        tmp = 'temporary-' + path
        mx.save_safetensors(tmp, data)
        os.replace(tmp, path)

    def load(self, path: str):
        import os
        if not os.path.exists(path): return

        data, model, opts = mx.load(path), {}, {}
        
        for k, v in data.items():
            if k.startswith("m."): model[k[2:]] = v
            elif k.startswith("o."): opts[k[2:]] = v
            elif k.startswith("state."): self.layers[int(k.split('.')[1])].states = v
            elif k.startswith("decaytrace."): self.layers[int(k.split('.')[1])].decaytrace = v
            elif k.startswith("embedtrace."): self.layers[int(k.split('.')[1])].embedtrace = v
            
        if model: self.update(util.tree_unflatten(list(model.items())))
        if opts: self.optimizer.state = util.tree_unflatten(list(opts.items()))

class Runtime:
    def __init__(self, path: str, threshold: float, **kwargs):
        self.model = Model(**kwargs)
        self.path = path
        self.threshold = threshold

        self.step = 0
        self.prevtime = None

    def save(self):
        self.step += 1
        if self.step % 500 == 0: self.model.save(self.path)

    def call(self, c: int, n: int | None, end: bool, readonly: bool = False, notrace: bool = False):
        outputs = self.model(c, n, end, notrace)
        if not readonly: self.save()
        return outputs

    def write(self, b: int):
        import sys
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, readonly: bool = False, notrace: bool = False):
        import itertools, time

        while True:
            text = input(f'\n[{self.now()} | {0 if self.prevtime is None else time.time() - self.prevtime:.4f}s]\nUser >> ')
            self.prevtime = time.time()

            data = (text + '\n').encode('utf-8')
            
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                b, _ = self.call(c, n, i == len(data) - 2, readonly, notrace)

            print(f'\n[{self.now()}]\nModel >> ', end = '', flush = True)

            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, readonly, notrace)
                self.write(b)
                if stop > self.threshold:
                    print()
                    break

    def dataset(self):
        import glob, itertools, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        files = glob.glob(os.path.join(root, 'wikipedia_clean/**/wiki_*'), recursive = True)

        while True:
            for file in files:
                with open(file, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                    for line in f:
                        data = line.encode('utf-8')
                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2)
                            self.write(b)

    def now(self):
        from datetime import datetime
        return datetime.now().strftime('%d/%m/%Y, %H:%M:%S')

    def __call__(self):
        modes = ['train', 'chat', 'chatreadonly', 'chatnotrace']

        try: mode = modes.index(input(f'\nthe \'chatnotrace\' mode is there for bug testing. \'chatreadonly\' is for chatting without overriding weights.\n[{self.now()}]\nmode: {modes} >> ').lower())
        except ValueError:
            print('\nInvalid mode.')
            return

        self.model.load(self.path)
        print()

        try:
            match mode:
                case 0: self.dataset()
                case 1: self.chat()
                case 2: self.chat(readonly = True)
                case 3: self.chat(readonly = True, notrace = True)

        finally:
            if mode < 2: self.model.save(self.path)

if __name__ == '__main__':
    # Runtime(path = 'larger-130m.safetensors', threshold = 0.35, dim = 2048, layers = 32, temp = 0.75, lr = 5e-4)()
    # FIXME: 'experimental-4.5m.safetensors' is not shipped with this handoff (nor produced by any script here).
    # The checkpoints actually used are base-10m*.safetensors in checkpoints/ (dim=768, layers=16). Point this at one of those.
    Runtime(path = 'experimental-4.5m.safetensors', threshold = 0.35, dim = 512, layers = 16, temp = 0.75, lr = 5e-4)()
    # param count = (256 * dim) + (dim * dim + dim * 2 + dim) + (256 * dim + dim + 1)
