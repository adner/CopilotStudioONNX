# tiny-storyteller — provenance & attribution

This skill bundles a third-party model. Credit and licensing:

## Model

- **Bundled files:** `model.onnx` (~2.6 MB, fp32 ONNX), `tokenizer.json`, `config.json`,
  `generation_config.json`.
- **ONNX export source:** [`onnx-community/TinyStories-656K-ONNX`](https://huggingface.co/onnx-community/TinyStories-656K-ONNX)
  on Hugging Face — **no license declared** on that repo (no model card as of 2024-11-22).
- **Original models:** the **TinyStories** family (`roneneldan/TinyStories-*`), introduced in
  *"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"*
  (Eldan & Li, 2023) — **no license declared** on the model repos.
- **Training dataset:** [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
  — licensed **CDLA-Sharing-1.0** (Community Data License Agreement – Sharing).

## Architecture (from `config.json`)

Llama (`LlamaForCausalLM`): 2 layers, 8 attention heads / 4 KV heads (grouped-query attention),
hidden size 128, head dim 16, vocab 2048, context 512, `bos = <|start_story|> = 1`,
`eos = <|end_story|> = 2`. ~656K parameters.

## Licensing summary

| Component | License |
|-----------|---------|
| `onnx-community/TinyStories-656K-ONNX` (ONNX export) | No license declared |
| `roneneldan/TinyStories-*` (original model weights) | No license declared |
| `roneneldan/TinyStories` (training dataset) | CDLA-Sharing-1.0 |

**Note:** The absence of an explicit model license, combined with CDLA-Sharing-1.0 on the
training data, creates ambiguity for redistribution. For this personal/demo use in a sandbox
there is no issue. For any broader distribution, treat the weights as CDLA-Sharing-1.0 by
inheritance and verify current terms on the Hugging Face pages above.

This package is a demonstration of running a very small LLM offline via ONNX Runtime; it adds no
additional model training and ships the weights unmodified from the upstream ONNX export.
