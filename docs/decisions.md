# Technical Decisions

## D001 - Target device: Jetson Orin NX 16GB

### Context

The project needs a real edge platform for:

- VLM deployment
- TensorRT
- C++ runtime development
- quantization
- profiling
- memory and power benchmarking

### Options Considered

1. Jetson Orin Nano Super 8GB
2. Jetson Orin NX 16GB
3. Jetson AGX Orin 32GB

### Decision

Use **Jetson Orin NX 16GB** as the primary deployment and benchmark device.

### Reasons

- 16GB unified memory gives a realistic chance to run a complete FP16 VLM baseline.
- FP16 / INT8 / INT4 can be compared on the same device.
- It still has meaningful edge-device memory and power constraints.
- It is more suitable than the 8GB version for full profiling and ablation.
- AGX Orin 32GB is more expensive and exceeds the main needs of the current 2B–4B VLM project.

### Trade-offs

- Orin is Ampere, so no native FP8 runtime path.
- 4B FP16 TensorRT engine build may still be memory-constrained.
- CPU, GPU, OS, weights, activations, KV cache and runtime all share the same 16GB memory.

### Status

Accepted.

---

## D002 - Primary model candidate: Qwen3-VL-4B-Instruct

### Context

The model should be large enough to make quantization meaningful, but still have a reasonable chance of supporting a complete FP16 baseline on Orin NX 16GB.

### Options Considered

- Qwen3-VL-2B-Instruct
- Qwen3-VL-4B-Instruct
- Qwen2.5-VL-3B-Instruct

### Decision

Try **Qwen3-VL-4B-Instruct** first.

Use **Qwen3-VL-2B-Instruct** as the development / fallback model.

### Reasons

- Qwen3-VL is newer than Qwen2.5-VL.
- 4B is large enough to make INT4 memory and performance gains meaningful.
- It is still within a plausible engineering range for Orin NX 16GB.
- The model is representative enough for a resume-quality deployment project.

### Re-evaluation Conditions

Move the full controlled experiment to 2B if 4B:

- repeatedly fails FP16 TensorRT engine build due to OOM;
- requires swap to run reliably;
- cannot use the same benchmark configuration as quantized variants;
- leaves too little memory for basic profiling;
- causes the project to spend most time solving OOM rather than quantization/runtime optimization.

If that happens, 4B remains useful as an INT4 deployment extension.

### Status

Pending FP16 engine feasibility test.

---

## D003 - Separate accuracy and performance baselines

### Accuracy Baseline

Use the original PyTorch BF16/FP16 model on an x86 rental GPU.

Purpose:

- quality reference;
- output regression;
- embedding/logit comparison;
- measuring accuracy loss caused by quantization.

### Performance Baseline

Use **TensorRT FP16 on the same Jetson Orin NX 16GB**.

Purpose:

- latency comparison;
- TTFT;
- decode throughput;
- peak memory;
- power;
- temperature;
- energy/request.

### Important Rule

Do not directly compare A100/4090 PyTorch speed with Jetson INT4 speed.

---

## D004 - Core quantization ablation

The four main experiment configurations:

| Config | Vision Encoder | Projector | LLM Backbone |
|---|---|---|---|
| A | FP16 | FP16 | FP16 |
| B | FP16 | FP16 | INT4 AWQ |
| C | custom INT8 | FP16 | FP16 |
| D | custom INT8 | FP16 | INT4 AWQ |

Initial final engineering candidate:

```text
Vision Encoder: mixed INT8
Projector: FP16
LLM Backbone: INT4 AWQ
KV Cache: FP16
Batch: 1
```

### Reasoning

This design lets the project separately measure:

- contribution from LLM INT4;
- contribution from Vision INT8;
- combined gain;
- which module causes quality loss;
- whether the bottleneck moves after quantization.

---

## D005 - Build TensorRT engines on Jetson

### Decision

Use x86 GPU for:

- PyTorch baseline;
- quantization;
- ONNX export.

Use Jetson for:

- TensorRT engine build;
- C++ runtime;
- final benchmark.

### Pipeline

```text
Hugging Face / PyTorch checkpoint
        ↓
x86 GPU: baseline / quantization / export
        ↓
ONNX
        ↓
Jetson Orin NX
        ↓
TensorRT engine build
        ↓
C++ runtime / benchmark
```

### Reason

TensorRT engine is tied to target hardware/platform/runtime characteristics. An x86 A100/4090 engine is not the deployment engine for ARM64 Orin.

---

## D006 - Dataset sampling must be reproducible

### Decision

Formal calibration and evaluation data will use:

- fixed random seed;
- preferably stratified random sampling;
- committed manifests.

Suggested seed:

```text
20260808
```

Manual selection is allowed only for:

- smoke tests;
- intentionally constructed edge cases;
- failure-case analysis.

### Reason

Avoid cherry-picking and ensure FP16 / INT8 / INT4 comparisons use exactly the same samples.

---

## D007 - Dataset roles are separated

### Vision INT8 Calibration

Use COCO `train2017`.

Initial target:

```text
512 images
```

Recommended nested calibration subsets:

```text
128 ⊂ 256 ⊂ 512 ⊂ 1024
```

### VQA / Reasoning Evaluation

Use GQA Balanced.

### Grounding Evaluation

Use RefCOCO+ / RefCOCOg first.

### Hallucination Evaluation

POPE is optional but useful.

### Performance Benchmark

Use a fixed representative set grouped by visual-token workload rather than a tiny purely random set.

---

## D008 - Original model should primarily live on laptop / x86 preparation environment

### Decision

The full Hugging Face checkpoint should be downloaded and kept primarily on the laptop or x86 rental/persistent storage.

Jetson should mainly hold:

- ONNX;
- tokenizer / processor files needed at runtime;
- TensorRT engines;
- benchmark data.

### Reason

The original checkpoint is mainly needed for:

- PyTorch accuracy baseline;
- quantization;
- export.

Keeping all large intermediate artifacts on the 256GB Jetson NVMe would waste target-device storage.

---

## D009 - Use sshuttle instead of per-application SOCKS configuration

### Decision

Prefer transparent routing through `sshuttle` for normal Jetson development traffic.

### Reason

Per-application proxy configuration created separate issues for:

- apt;
- git;
- curl;
- hf;
- pip.

A network-layer TCP/DNS tunnel is easier to maintain for the development workflow.
