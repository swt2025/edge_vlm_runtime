# Experiments

This file records experiments that were actually executed.

Recommended entry format:

```text
Date
Goal
Environment
Input / Configuration
Command
Result
Interpretation
Next Step
```

---

## E001 - TensorRT small-engine smoke test

### Date

2026-08-04

### Goal

Verify the basic Jetson TensorRT runtime path:

```text
TensorRT Engine
→ deserialize
→ allocate buffers
→ H2D
→ GPU inference
→ D2H
```

### Environment

- Jetson Orin NX 16GB
- Ubuntu 22.04.5
- L4T R36.4.3
- CUDA 12.6
- TensorRT 10.3
- Power mode: MAXN_SUPER

### Command

A small FP16 TensorRT engine was loaded using:

```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=/home/nvidia/edge_vlm_runtime/results/device_baseline/tensorrt/mnist_fp16.engine \
  --warmUp=500 \
  --duration=10
```

### Result

Observed:

```text
Throughput: 13717.5 qps
Latency mean: 0.063977 ms
GPU Compute Time mean: 0.0507263 ms
Total Host Walltime: 10.0002 s
Total GPU Compute Time: 6.95849 s
```

TensorRT reported:

```text
&&&& PASSED TensorRT.trtexec
```

A warning was observed:

```text
GPU compute time is unstable, coefficient of variance = 6.36899%
```

### Interpretation

The test confirms:

- TensorRT runtime works;
- engine deserialization works;
- CUDA / TensorRT execution works;
- the basic GPU inference path is functional.

The throughput/latency values are **not** representative of VLM performance because the network is tiny and fixed overhead dominates.

### Next Step

For formal VLM benchmarks:

- use a realistic model;
- record power/frequency state;
- warm up consistently;
- record `tegrastats`;
- consider `jetson_clocks`;
- use fixed benchmark inputs.

---

## Planned Experiment - Qwen3-VL-4B FP16 feasibility gate

### Goal

Decide whether Qwen3-VL-4B can remain the full experimental model on Orin NX 16GB.

### Initial Controlled Configuration

```text
batch_size = 1
max_images = 1
visual_tokens = 256
max_input_length = 512 or 1024
max_kv_cache_capacity = 2048
max_new_tokens = 64
```

### Phase 1 - Engine Build

Build separately:

1. Visual FP16 engine
2. LLM FP16 engine

Record:

- build success/failure;
- peak memory;
- build time;
- workspace limit;
- builder optimization level;
- swap usage;
- engine size.

### Phase 2 - Stability

- warm up 10 times;
- run 50 measured requests;
- monitor RAM, swap, temperature, frequency and power;
- check latency stability.

### Phase 3 - Increase Workload

Increase one variable at a time:

```text
visual_tokens: 256 → 512
KV capacity: 2048 → 4096
output length: 64 → 128
```

### Pass Criteria

4B becomes the full project model if:

- FP16 engine builds successfully;
- runtime does not depend on swap;
- repeated inference is stable;
- basic profiling remains possible;
- FP16 and quantized configurations can use the same controlled workload.

Otherwise use Qwen3-VL-2B for the complete A/B/C/D ablation and keep 4B as a quantized deployment extension.

---

## Planned Experiment - Core precision ablation

| Config | Vision | Projector | LLM |
|---|---|---|---|
| A | FP16 | FP16 | FP16 |
| B | FP16 | FP16 | INT4 AWQ |
| C | INT8 | FP16 | FP16 |
| D | INT8 | FP16 | INT4 AWQ |

Measure:

- quality / accuracy;
- Vision latency;
- Prefill latency;
- TTFT;
- Decode tokens/s;
- TPOT;
- end-to-end latency;
- peak memory;
- average power;
- energy/request;
- temperature;
- throttling.

---

## Planned Experiment - Vision INT8 calibration set size

Use nested COCO calibration sets:

```text
128
256
512
1024
```

Measure whether additional calibration images continue to improve:

- visual embedding cosine similarity;
- task accuracy;
- grounding IoU;
- final VLM quality.

Goal:

identify the point where additional calibration data gives little benefit.
