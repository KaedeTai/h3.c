#!/usr/bin/env python3
"""Run the upstream typed_decisions trainer with the Chinese augmentation swapped in.

Nothing in the trainer changes: train_encoder imports `augment` and `pair` by name, so
rebinding those two names in its module is the whole difference from upstream.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "jev_eval", "typed-decisions-src", "src"))

import augment_zh  # noqa: E402
from typed_decisions import train_encoder  # noqa: E402

train_encoder.augment = augment_zh.augment
train_encoder.pair = augment_zh.pair


def pad_to_buckets(step=64):
    """Round every batch's length up to a multiple of `step`.

    MPS compiles kernels per tensor shape. With dynamic padding every batch is a new length and
    a new compile: measured 0.7 s/step for a shape already seen against 6-12 s for a fresh one,
    so an unbucketed epoch spends most of its time compiling. Padding is masked and pooled into
    the junk slot, so the outputs are unchanged.
    """
    import torch.nn.functional as F
    from typed_decisions import encoder as enc
    orig = enc.Collator.__call__

    def call(self, items, device=None):
        out = orig(self, items, None)
        L = out["input_ids"].shape[1]
        Lp = min(self.max_len, -(-L // step) * step)
        if Lp > L:
            pad = Lp - L
            junk = int(out["seg"].max())        # CLS is never in a span, so the max is always the junk slot
            out["input_ids"] = F.pad(out["input_ids"], (0, pad), value=self.tok.pad_token_id)
            out["attention_mask"] = F.pad(out["attention_mask"], (0, pad), value=0)
            out["seg"] = F.pad(out["seg"], (0, pad), value=junk)
        return {k: (v.to(device) if device is not None else v) for k, v in out.items()}

    enc.Collator.__call__ = call


pad_to_buckets()


class BucketedRandom:
    """Stands in for the `random` module inside train_encoder and changes only shuffle().

    The trainer shuffles uniformly, so nearly every batch of 16 holds one MASSIVE state with a
    60-option question (~420 tokens) and the other fifteen, ~80-token caption states, are padded
    to match. Sorting by length within windows of 32 batches, then shuffling whole batches, keeps
    the epoch random at the batch level and stops paying for padding. Every other call
    (random(), choice(), seed()...) passes straight through, so augmentation is untouched.
    """

    def __init__(self, lengths, batch, window_batches=32):
        import random as _r
        self._r, self.lengths, self.batch, self.window = _r, lengths, batch, batch * window_batches

    def shuffle(self, x):
        self._r.shuffle(x)
        if len(x) != len(self.lengths):
            return                                  # not the epoch order: leave it alone
        flat = []
        for i in range(0, len(x), self.window):
            flat += sorted(x[i:i + self.window], key=lambda j: self.lengths[j])
        batches = [flat[i:i + self.batch] for i in range(0, len(flat), self.batch)]
        self._r.shuffle(batches)
        x[:] = [j for b in batches for j in b]

    def __getattr__(self, k):
        return getattr(self._r, k)


def _lengths(data_dir, limit):
    import json
    L = []
    with open(os.path.join(data_dir, "train.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n = min(len(d["state"]), 256) + sum(2 + len(q["instructions"]) + sum(1 + len(o) for o in q["options"])
                                               for q in d["questions"])
            L.append(n)
    return L[:limit] if limit else L


def _arg(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--no-bucket" in argv:
        argv.remove("--no-bucket")
    else:
        lengths = _lengths(_arg(argv, "--data", "data"), int(_arg(argv, "--limit", "0")))
        train_encoder.random = BucketedRandom(lengths, int(_arg(argv, "--batch", "16")))
    train_encoder.main(argv)
