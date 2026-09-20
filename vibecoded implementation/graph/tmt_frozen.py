"""
Shared frozen-TMT forward + multiple-choice scoring helpers, used by build_graph_tensors.py,
train_hooks.py and eval_graph.py (archived in archive_madeup/). Same manual per-layer loop pattern as benchmark.py
everywhere in this project: no calls into Model.__call__'s training path, so TTT is off, but
state/trace still persists across bytes within a run (hard rules 5 and 6). No changes to
main.py; this only calls into Model's existing public pieces.

Updated for the upstream trace-topology fix (origin/main 94a6064, the fix discussed in issue #2):
`Layer.__call__` is now `(enc, x, dummy)` -- every layer's recurrent state receives the raw
per-step embedding directly instead of the previous layer's post-nonlinearity output -- and
`embedtrace` moved from one shared `Encoder.embedtrace` buffer to one per `Layer`. This module's
own `reset_state()` deliberately does NOT use the new `Model.reset()` from main.py: that method
also zeroes `layer.decay`, which is a *trainable* parameter (the layer's learned time constant),
not per-sequence memory -- zeroing it on every reset would silently discard the checkpoint's
learned decay rates every single item. Only `states`/`decaytrace`/`embedtrace` are genuinely
per-sequence memory and get cleared here; `decay` is left untouched, as it always was before the
upstream change existed.
"""
import mlx.core as mx


def reset_state(model, dim):
    for layer in model.layers:
        layer.states = mx.zeros((dim,))
        layer.decaytrace = mx.zeros((dim,))
        layer.embedtrace = mx.zeros((256, dim))


def snapshot_state(model):
    return (
        [mx.array(layer.states) for layer in model.layers],
        [mx.array(layer.decaytrace) for layer in model.layers],
        [mx.array(layer.embedtrace) for layer in model.layers],
    )


def restore_state(model, snap):
    states, decaytraces, embedtraces = snap
    for layer, s, d, e in zip(model.layers, states, decaytraces, embedtraces):
        layer.states = s
        layer.decaytrace = d
        layer.embedtrace = e


def step(model, dim, dummies, c, hooks=None, gt=None, query_z=None):
    """One frozen forward step for byte c. If hooks+gt (graph tensors dict with K/A/F/M) are
    given, applies Stage A before the decoder. `query_z`, if given, overrides what gets
    projected into the search query (see graph_hooks.GraphHooks.retrieve's query_z docstring --
    the "query anchor" fix for identity decay). Returns (logits, stop, hook_log, x_before_hook)
    -- x_before_hook lets a caller snapshot this exact step's latent to use as a later anchor."""
    enc = model.encoder(mx.array(c))
    x = enc
    for j, layer in enumerate(model.layers):
        x, state, decay = layer(enc, x, dummies[j])
        layer.states = mx.stop_gradient(state)
    x_before_hook = x
    hook_log = None
    if hooks is not None and gt is not None:
        x, hook_log = hooks.apply_stage_a(x, gt["K"], gt["A"], gt["F"], gt["M"], query_z=query_z)
    logits, stop = model.decoder(x)
    return logits, stop, hook_log, x_before_hook


def embed_text(model, dim, layers, text: str) -> mx.array:
    """Resets state, runs `text` through the frozen model byte-by-byte, returns the final
    per-step latent (x after the last layer, before the decoder) -- used offline to build node
    signposts (K) and fact vectors (F). No hooks involved; this is what build_graph_tensors.py
    uses to make the tensors graph_hooks.py later reads at runtime."""
    reset_state(model, dim)
    dummies = [mx.zeros((dim,)) for _ in range(layers)]
    x_before_hook = None
    for b in text.encode("utf-8"):
        _, _, _, x_before_hook = step(model, dim, dummies, b)
    return x_before_hook


def score_candidates(model, dim, layers, prompt_bytes, candidates, hooks=None, gt=None,
                      anchor_byte_idx=None):
    """Teacher-forced total log-probability of each candidate string continuing prompt_bytes.
    Resets state, runs the shared prompt once, snapshots state right after it, then rewinds to
    that snapshot before scoring each candidate independently (so candidates never contaminate
    each other's state). Returns a list of python floats, same order as `candidates`.

    anchor_byte_idx: if given, the latent at prompt_bytes[anchor_byte_idx] (0-indexed) is
    captured and used as the query anchor (query_z) for every subsequent step, both the rest of
    the prompt and all candidate scoring -- see graph_hooks.py's query_z docstring for why.
    """
    reset_state(model, dim)
    dummies = [mx.zeros((dim,)) for _ in range(layers)]
    last_logits = None
    anchor = None
    for idx, b in enumerate(prompt_bytes):
        last_logits, _, _, x_raw = step(model, dim, dummies, b, hooks, gt, query_z=anchor)
        if anchor_byte_idx is not None and idx == anchor_byte_idx:
            anchor = mx.stop_gradient(x_raw)
    snap = snapshot_state(model)

    scores = []
    for cand in candidates:
        restore_state(model, snap)
        cand_dummies = [mx.zeros((dim,)) for _ in range(layers)]
        cand_bytes = cand.encode("utf-8")

        logp = mx.log(mx.softmax(last_logits)[cand_bytes[0]] + 1e-12)
        cur_logits = last_logits
        for i in range(len(cand_bytes) - 1):
            cur_logits, _, _, _ = step(model, dim, cand_dummies, cand_bytes[i], hooks, gt, query_z=anchor)
            logp = logp + mx.log(mx.softmax(cur_logits)[cand_bytes[i + 1]] + 1e-12)
        scores.append(logp.item())

    restore_state(model, snap)  # leave model state clean for the caller
    return scores
