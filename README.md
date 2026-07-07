# MyAILocalModel

MyAILocalModel is a local AI model advisor and launcher. It scans your machine, searches Hugging Face for local model artifacts, explains why one model is a better fit than another, downloads the selected model, and prepares a local runtime profile.

The goal is not to make users learn GGUF, VRAM, quantization, llama.cpp, Ollama, or runtime flags before they can try local AI. The app should explain the tradeoffs in plain English and make the safe path obvious.

## Current build

This repository contains a Tauri 2 + Rust + React/TypeScript desktop application foundation with:

- hardware scan view;
- Hugging Face GGUF model search;
- explainable recommendation scoring;
- speed/quality preference slider;
- task-based recommendation hints;
- model comparison cards;
- quantization explanation;
- download flow with progress state in the UI;
- runtime guidance for llama.cpp, Ollama, LM Studio, vLLM, ExLlama, MLC, and TensorRT-LLM;
- Rust unit tests for scoring behavior.

## Run locally

Install Node.js, Rust, and Tauri prerequisites, then:

```bash
npm install
npm run tauri dev
```

Run tests:

```bash
cargo test --manifest-path src-tauri/Cargo.toml
npm run typecheck
```

## Environment

Optional Hugging Face token for gated models:

```bash
HF_TOKEN=hf_...
```

Downloaded models are stored under the operating system app data directory by default.

## Product philosophy

The advisor should tell the user:

- what their machine can actually run;
- which models fit comfortably;
- which models are possible but risky;
- why a smaller quantized model may be better than a larger one;
- when GPU offload is realistic;
- when CPU-only inference will be slow but usable;
- which runtime makes the most sense for their hardware and goal.

This should become the easiest on-ramp to local AI for normal users and small businesses that want privacy without terminal work.