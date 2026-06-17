#!/usr/bin/env python3
"""
tiny-storyteller: run a *tiny* 656K-parameter TinyStories LLM (ONNX) fully offline.

Uses ONLY onnxruntime + tokenizers + numpy -- no transformers / optimum / torch /
onnxruntime-genai. The text-generation loop is hand-written.

The bundled model (model/model.onnx) is a 656K-param Llama trained on the TinyStories
dataset, exported with a KV-cache ("with past") signature:
    inputs : input_ids, attention_mask, position_ids, past_key_values.{i}.{key,value}
    outputs: logits, present.{i}.{key,value}
We drive it *without* a KV cache: every step we feed the full token sequence plus
ZERO-LENGTH past tensors, so the model recomputes the whole sequence. That's O(n^2),
but at 656K params and short stories it's instant -- and it keeps the loop trivial.

Crucially, the past tensors are shaped from each input's *declared* shape (this model
uses grouped-query attention: 4 KV heads, head_dim 16), not from a head count guessed
out of config.json.

All bundled files (this script + model.onnx + tokenizer.json + config.json) sit in ONE
flat directory, and the model files are located via a tolerant search (see find_asset),
so this works even if the skill host renames/flattens the bundle.

Examples
--------
  python generate.py --selftest
  python generate.py --prompt "Once upon a time, a little cat" --max-new-tokens 120
  python generate.py --prompt "Lily found a red ball" --greedy --max-new-tokens 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def find_asset(*names):
    """Locate a bundled asset robustly, whatever layout the host used.

    Some skill hosts (e.g. Copilot Studio) FLATTEN the uploaded bundle: a file that
    was at `model/model.onnx` ends up next to this script as `model__model.onnx`.
    Others keep a `model/` subfolder, or drop everything in one directory. We try the
    common spots and name variants, then fall back to a recursive search by basename.
    """
    dirs = []
    for base in (SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd()):
        dirs += [base, base / "model"]
    seen = set()
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)
        for n in names:
            p = d / n
            if p.is_file():
                return p
    for root in (SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd()):
        for n in names:
            for hit in root.rglob(n):
                if hit.is_file():
                    return hit
    sys.exit(f"ERROR: could not find any of {names} near {SCRIPT_DIR}")


def log(*args):
    """Status goes to stderr so stdout stays clean (= just the generated text)."""
    print(*args, file=sys.stderr, flush=True)


def load_config():
    with open(find_asset("config.json", "model__config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer():
    return Tokenizer.from_file(str(find_asset("tokenizer.json", "model__tokenizer.json")))


def build_session():
    path = find_asset("model.onnx", "model__model.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so,
                                providers=["CPUExecutionProvider"])


def _zero_past_shape(declared_shape):
    """Turn a declared past-tensor shape into a concrete zero-length shape.

    Declared dims are a mix of ints (fixed, e.g. num_kv_heads=4, head_dim=16) and
    strings (symbolic, e.g. 'batch_size', 'past_sequence_length'). We keep ints as-is,
    map a batch-like symbolic dim to 1, and any other symbolic dim (the past length) to 0.
    """
    out = []
    for d in declared_shape:
        if isinstance(d, int):
            out.append(d)
        elif isinstance(d, str) and "batch" in d.lower():
            out.append(1)
        else:
            out.append(0)  # past_sequence_length -> empty cache
    return tuple(out)


def make_feed(session, input_ids):
    """Build the feed dict adaptively from the model's declared inputs (no-cache mode)."""
    ids = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
    seq = ids.shape[1]
    feed = {}
    for inp in session.get_inputs():
        name = inp.name
        if name == "input_ids":
            feed[name] = ids
        elif name == "attention_mask":
            feed[name] = np.ones((1, seq), dtype=np.int64)
        elif name == "position_ids":
            feed[name] = np.arange(seq, dtype=np.int64).reshape(1, seq)
        elif name == "use_cache_branch":
            feed[name] = np.array([False])
        elif name.startswith("past_key_values") or name.startswith("past"):
            feed[name] = np.zeros(_zero_past_shape(inp.shape), dtype=np.float32)
        else:
            sys.exit(f"ERROR: model wants an input {name!r} this script does not know "
                     f"how to provide (shape={inp.shape}, type={inp.type}).")
    return feed


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def pick_next(logits, *, greedy, temperature, top_k, top_p, rng):
    """Choose the next token id from the last-position logits (1-D array)."""
    if greedy or temperature <= 0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64) / temperature

    if top_k and top_k > 0:
        top_k = min(top_k, logits.shape[0])
        kth = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[kth] = logits[kth]
        logits = mask

    probs = softmax(logits)

    if top_p and 0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cum = np.cumsum(probs[order])
        keep = order[: max(1, int(np.searchsorted(cum, top_p)) + 1)]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        s = filtered.sum()
        probs = filtered / s if s > 0 else probs

    return int(rng.choice(len(probs), p=probs))


def encode_prompt(tok, prompt, bos_id):
    ids = tok.encode(prompt).ids if prompt else []
    # Llama/TinyStories expect a leading BOS; add it if the tokenizer didn't.
    if bos_id is not None and (not ids or ids[0] != bos_id):
        ids = [bos_id] + ids
    return ids


def generate(prompt, *, max_new_tokens, greedy, temperature, top_k, top_p, seed):
    cfg = load_config()
    n_ctx = cfg.get("max_position_embeddings") or 512
    eos_id = cfg.get("eos_token_id", 2)
    bos_id = cfg.get("bos_token_id", 1)

    tok = load_tokenizer()
    session = build_session()
    rng = np.random.default_rng(seed)

    ids = encode_prompt(tok, prompt, bos_id)
    prompt_len = len(ids)
    if prompt_len >= n_ctx:
        sys.exit(f"ERROR: prompt is {prompt_len} tokens; context limit is {n_ctx}.")

    budget = min(max_new_tokens, n_ctx - prompt_len)
    log(f"[tiny-storyteller] prompt={prompt_len} tok, generating up to {budget} "
        f"({'greedy' if greedy else f'temp={temperature} top_k={top_k} top_p={top_p}'})")

    t0 = time.time()
    new = 0
    for _ in range(budget):
        logits = session.run(["logits"], make_feed(session, ids))[0]  # [1, seq, vocab]
        nxt = pick_next(logits[0, -1, :], greedy=greedy, temperature=temperature,
                        top_k=top_k, top_p=top_p, rng=rng)
        if nxt == eos_id:
            break
        ids.append(nxt)
        new += 1

    dt = time.time() - t0
    log(f"[tiny-storyteller] {new} tokens in {dt:.1f}s "
        f"({new/dt:.1f} tok/s)" if dt > 0 else f"[tiny-storyteller] {new} tokens")

    # Drop the leading BOS before decoding so it doesn't surface as a stray token.
    out_ids = ids[1:] if (bos_id is not None and ids and ids[0] == bos_id) else ids
    text = tok.decode(out_ids)
    # The model sometimes *spells out* the story delimiters as ordinary tokens
    # instead of emitting the single special token; trim anything from there on.
    for marker in ("<|end_story|>", "<|start_story|>", "<|endoftext|>"):
        text = text.split(marker)[0]
    return text.strip()


def selftest():
    """One forward pass; verify the runtime works and the logits shape is sane."""
    cfg = load_config()
    vocab = cfg.get("vocab_size", 2048)
    bos_id = cfg.get("bos_token_id", 1)
    tok = load_tokenizer()
    session = build_session()
    ids = encode_prompt(tok, "Once upon a time", bos_id)
    logits = session.run(["logits"], make_feed(session, ids))[0]
    assert logits.shape == (1, len(ids), vocab), \
        f"unexpected logits shape {logits.shape}, expected (1, {len(ids)}, {vocab})"
    nxt = int(np.argmax(logits[0, -1, :]))
    log(f"[selftest] OK -- logits {logits.shape}, greedy next id={nxt} "
        f"({tok.decode([nxt])!r})")
    print("SELFTEST PASSED")


def main(argv=None):
    p = argparse.ArgumentParser(description="Run a 656K TinyStories LLM (ONNX) offline.")
    p.add_argument("--prompt", default="Once upon a time", help="text to continue")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--greedy", action="store_true", help="deterministic argmax decoding")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true",
                   help="run a single forward pass and exit")
    args = p.parse_args(argv)

    if args.selftest:
        selftest()
        return

    print(generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
