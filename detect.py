import json, re, sys, statistics
import requests, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GIST = sys.argv[1] if len(sys.argv) > 1 else "https://api.github.com/gists/f690e437e7f28042776ed00015091bd9"
MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"

g = requests.get(GIST, headers={"User-Agent":"paper-lint","Accept":"application/vnd.github+json"}, timeout=60).json()
files = g["files"]
raw = files["ARC-2026-paper-draft.md"]["raw_url"] if "ARC-2026-paper-draft.md" in files else list(files.values())[0]["raw_url"]
text = requests.get(raw, headers={"User-Agent":"paper-lint"}, timeout=60).text
print("DOC_CHARS: " + str(len(text)), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
print("ID2LABEL: " + json.dumps(model.config.id2label), flush=True)

def score(t):
    ids = tok(t, return_tensors="pt", truncation=False)["input_ids"][0]
    n = 512
    windows = [ids] if len(ids) <= n else [ids[i:i+n] for i in range(0, len(ids), n//2)][:8]
    probs = []
    with torch.no_grad():
        for w in windows:
            out = model(input_ids=w.unsqueeze(0), attention_mask=torch.ones(1, len(w)))
            probs.append(torch.softmax(out.logits, dim=-1)[0].tolist())
    return [sum(c)/len(c) for c in zip(*probs)], len(windows)

parts = re.split(r"(?m)^## ", text)
results = {}
for p in parts:
    p = p.strip()
    if len(p) < 200: continue
    title = p.split("\n", 1)[0][:48]
    avg, nw = score(p)
    results[title] = {"probs": [round(x, 4) for x in avg], "windows": nw, "chars": len(p)}
print("RESULTS_JSON:" + json.dumps(results), flush=True)
sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
lens = [len(s.split()) for s in sents]
print(f"HEUR em_dashes={text.count(chr(0x2014))} sent_mean={statistics.mean(lens):.1f} sent_stdev={statistics.pstdev(lens):.1f} nsent={len(lens)}", flush=True)
