# More Fun With New Copilot Studio

Exploring the new [**Copilot Studio Orchestrator**](https://learn.microsoft.com/en-us/microsoft-copilot-studio/agents-experience/overview) — which spins up a sandboxed Azure Linux
container per agent session — by building uploadable **Agent Skill packages** that run real
AI models fully offline using the Python packages pre-installed in the sandbox. Check out [this LinkedIn post](https://www.linkedin.com/posts/andreasadner_copilotstudio-powerplatform-onnxruntime-ugcPost-7472674468692353024-1fIW) for a demo.

## Skills

### `tiny-storyteller`

A 656K-parameter TinyStories LLM in ONNX format, running fully offline on-box via
`onnxruntime`. Ask the agent to write a short children's story and it will run a real (tiny,
charmingly limited) language model right there in the sandbox — no internet, no API call.

- **Model:** `onnx-community/TinyStories-656K-ONNX` — 2.6 MB, Llama architecture, vocab 2048
- **Runtime deps (pre-installed in sandbox):** `onnxruntime`, `tokenizers`, `numpy`
- **Package size:** ~2.7 MB (1.2 MB zipped)

---

## Setup

### Prerequisites

- Python 3.8+ (for local testing)
- `onnxruntime`, `tokenizers`, `numpy` installed locally:
  ```
  pip install onnxruntime tokenizers numpy
  ```

### 1. Clone the repo

```bash
git clone https://github.com/adner/CopilotStudioONNX.git
cd MoreFunWithNewCopilotStudio
```

### 2. Download the model

The `model.onnx` file is not committed to git (no explicit license on the weights — see
[`tiny-storyteller/NOTICE.md`](tiny-storyteller/NOTICE.md)). Fetch it from Hugging Face:

```bash
python tiny-storyteller/download_model.py
```

This downloads `model.onnx` (~2.6 MB) into `tiny-storyteller/` from
`onnx-community/TinyStories-656K-ONNX` on Hugging Face. Requires internet access and no
extra packages (uses Python's built-in `urllib`).

### 3. Test locally

```bash
# Verify onnxruntime can load and run the model
python tiny-storyteller/generate.py --selftest

# Generate a story
python tiny-storyteller/generate.py --prompt "Once upon a time, a little cat" --max-new-tokens 120
```

### 4. Package for Copilot Studio

Zip the skill contents **flat** (all files at the archive root — Copilot Studio requires
`SKILL.md` to be at the root level):

```bash
# PowerShell
Compress-Archive -Path tiny-storyteller\* -DestinationPath tiny-storyteller.zip -CompressionLevel Optimal

# bash / zip
cd tiny-storyteller && zip -j ../tiny-storyteller.zip * && cd ..
```

Then upload `tiny-storyteller.zip` to your Copilot Studio agent.

### 5. Use the skill in Copilot Studio

Once uploaded, ask the agent something like:

> *"Use the tiny-storyteller skill to write a story about a puppy who finds a mysterious door."*

The agent will run `generate.py` on-box and return the generated story.

---

## How it works

The sandbox has no internet access at runtime, so everything must be bundled. It also ships
with a fixed set of Python packages — and while `onnxruntime` is included, the higher-level
libraries that normally make running an LLM straightforward are not:

| What you'd normally use | Available in sandbox? |
|---|---|
| `onnxruntime` | ✅ Yes |
| `tokenizers` (standalone HF tokenizer library) | ✅ Yes |
| `numpy` | ✅ Yes |
| `onnxruntime-genai` (Microsoft's high-level generate API) | ❌ No |
| `optimum` (Hugging Face ONNX export + inference) | ❌ No |
| `transformers` (Hugging Face model + tokenizer pipelines) | ❌ No |

That means there's no `pipeline()`, no `model.generate()`, no ready-made sampling loop — just
raw tensor in, tensor out. So `generate.py` hand-rolls everything: it tokenizes the prompt
using the standalone `tokenizers` library (loading `tokenizer.json` directly), builds the
numpy input arrays, calls `session.run()` in a loop, picks the next token from the logits
(greedy or top-k/top-p sampling), and decodes the result — all without any of the usual
Hugging Face abstractions.

Key details of the inference loop:
- **No KV cache** — the script feeds zero-length `past_key_values` tensors each step and
  recomputes the full sequence. Simple and fast enough at 656K params.
- **GQA-aware** — past tensor shapes are derived from the model's *declared* input shapes
  (4 KV heads, not 8), not from `config.json`.
- **Flat-bundle tolerant** — Copilot Studio flattens subfolders into `model__filename`
  prefixes; `find_asset()` in `generate.py` locates files under either naming scheme.

## Available Python packages in the sandbox

See [`pythonpackages.md`](pythonpackages.md) for the full list of packages pre-installed in
the Copilot Studio container.

## License

The skill code in this repo is unlicensed — do whatever you want with it. The bundled
**model weights** are a separate matter; see
[`tiny-storyteller/NOTICE.md`](tiny-storyteller/NOTICE.md) for details.
