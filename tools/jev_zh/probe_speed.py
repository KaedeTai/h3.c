import os, sys, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "jev_eval", "typed-decisions-src", "src"))
import torch
import train_zh  # installs bucket padding
from typed_decisions.schema import read_jsonl
from typed_decisions.encoder import DecisionEncoder, Collator, load_tokenizer, decision_loss
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/chinese-roberta-wwm-ext-large")
BS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
dev = torch.device("mps")
tok = load_tokenizer(MODEL)
m = DecisionEncoder.from_pretrained(MODEL, tok, attn_implementation="sdpa").to(dev)
coll = Collator(tok, 256, 512)
train = read_jsonl(os.path.join(HERE, "data", "train.jsonl"))
import train_zh
lens = train_zh._lengths(os.path.join(HERE, "data"), 0)
order = list(range(len(train)))
br = train_zh.BucketedRandom(lens, BS)
br.shuffle(order)
train = [train[j] for j in order[:BS * 12]]
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
m.train()
L = []
for s in range(12):
    b = coll([(e.state, e.questions) for e in train[s * BS:(s + 1) * BS]], dev)
    t0 = time.time()
    lg = m(b["input_ids"], b["attention_mask"], b["opt_pos"], b["opt_mask"], b["q_pos"], b["seg"])
    loss, _ = decision_loss(lg.float(), b["gold"])
    opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    dt = time.time() - t0
    L.append(dt)
    print(f"step {s} {dt:.2f}s  seq {b['input_ids'].shape[1]}  loss {loss.item():.3f}", flush=True)
print(f"steady median {sorted(L[2:])[len(L[2:]) // 2]:.2f}s/step at batch {BS}", flush=True)
