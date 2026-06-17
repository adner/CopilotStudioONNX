---
name: tiny-storyteller
description: "Write short, simple children's stories with a tiny 656K-parameter TinyStories language model bundled in ONNX format and run fully offline on-box with onnxruntime. Use this when the user wants to demo or test running a small local/internal LLM in the sandbox, generate or continue a little story from a prompt with no internet or external API, or verify that on-box ONNX text generation works. Keywords - local LLM, offline LLM, ONNX, onnxruntime, TinyStories, story generation, text generation, no internet."
---

# tiny-storyteller — a 656K-param offline LLM in ONNX

## Overview

This skill bundles a **TinyStories** model (~656K parameters, a tiny Llama architecture) as an
**ONNX** file and runs it **entirely on the box** with `onnxruntime`, `tokenizers`, and `numpy`
— packages already installed in the sandbox. **No internet, no API key, no external service.**
The model (2.6 MB) and tokenizer ship inside this package.

The model only knows how to do one thing: continue **short, simple, child-like stories** in
basic English (it was trained purely on the TinyStories dataset, vocabulary ~2048 tokens). It is
charmingly limited — expect simple sentences, repetition, and gentle nonsense. That's the point:
this is a demo of fully on-box LLM inference with a genuinely tiny model.

## When to use

Invoke this skill when the user asks to:
- run / try / demo a **small internal or local LLM** in the sandbox,
- **write or continue a little story** (especially a children's story) with no internet,
- **prove** that ONNX / `onnxruntime` text generation works on the box.

## How to run

The entry point is `generate.py`. All files in this skill (the script plus `model.onnx`,
`tokenizer.json`, `config.json`) live in the **same directory** — run the script from that
directory with the Python already on the box. (The script auto-locates its model files, so it
still works even if the host renamed or flattened the bundle, e.g. `model__model.onnx`.)

**1. (Recommended first) Verify the runtime works:**
```bash
python generate.py --selftest
```
Prints `SELFTEST PASSED` after one forward pass.

**2. Generate a story from a prompt:**
```bash
python generate.py --prompt "Once upon a time, a little cat" --max-new-tokens 120
```
The story is printed to **stdout**; status/progress lines go to **stderr**. Return the stdout
text to the user.

**3. Deterministic (reproducible) output:**
```bash
python generate.py --prompt "Lily found a big red ball" --greedy --max-new-tokens 100
```

If `python generate.py` can't find the script by name, locate it first (the host may have
flattened paths): `find / -name 'generate.py' 2>/dev/null` and run that path.

Tip: start prompts like a storybook ("Once upon a time…", "Tom and his dog…", "Lily found…").
The model was trained only on simple stories, so story-style prompts give the best results.

## Parameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--prompt` | `"Once upon a time"` | Story opening to continue. |
| `--max-new-tokens` | `120` | Max tokens to generate (capped at the 512-token context). |
| `--greedy` | off | Deterministic argmax decoding. More repetitive; good for reproducibility. |
| `--temperature` | `0.8` | Sampling randomness (used when not `--greedy`). Higher = wilder. |
| `--top-k` | `40` | Sample only from the top-k tokens. |
| `--top-p` | `0.95` | Nucleus sampling cutoff. |
| `--seed` | `42` | RNG seed; change it for a different story, keep it for reproducibility. |

A few ready-made prompts are in `prompts.txt`.

## Examples

```bash
python generate.py --prompt "Once upon a time, a little cat" --greedy --max-new-tokens 100
# -> Once upon a time, a little cat and a little mouse lived in a small house. They liked to
#    play together in the kitchen. One day, they saw a big tree with many leaves...

python generate.py --prompt "Lily found a big red ball in the park" --max-new-tokens 120 --seed 3
# -> Lily found a big red ball in the park. She was very happy. She wanted to play with it...
```

## Notes & limits

- **~656K params, CPU-only, simple English only.** Context window is 512 tokens.
- Output is intentionally tiny-model quality: simple sentences, repetition, occasional
  non-sequiturs, and characters that change name mid-story. That's expected and part of the demo.
- Fully offline — `model.onnx` (2.6 MB) and `tokenizer.json` are bundled alongside the script;
  nothing is downloaded at runtime.
- Very fast (hundreds–thousands of tokens/sec on CPU). The bundled graph has no KV cache (the
  script recomputes the full sequence each step), which is fine within the 512-token window.

## Troubleshooting

- **`ModuleNotFoundError`** for `onnxruntime` / `tokenizers` / `numpy`: these are preinstalled in
  the sandbox; just use the box's default `python`.
- **Story is repetitive or weird**: expected for a model this small. Drop `--greedy`, and/or
  raise `--temperature` (e.g. `1.0`).
- **Non-story prompts produce nonsense**: the model only knows simple stories — phrase the prompt
  as a story opening.
- **Model file missing**: `model.onnx` is not in the git repo — run `python download_model.py`
  from the `tiny-storyteller/` directory first, then re-zip and re-upload. The script also
  accepts a `model__model.onnx` name if the host flattened the bundle.
