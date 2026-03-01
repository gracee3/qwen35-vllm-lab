# qwen35-vllm-lab

## Current Triage Outcome (Decisive)

- FP8 startup failures are **not** KV/context-memory related.
  - We reduced `gpu-memory-utilization`, lowered batched prefill tokens, and used `--enforce-eager` while still hitting the same failure during model init.
- The crash consistently points to FP8 MoE fused weight allocation, not runtime cache growth:
  - `RuntimeError: ... fp8.py create_weights ... torch.empty`
  - `Tried to allocate ...` (around 512 MiB) during startup.
- Backend forcing confirmed:
  - `VLLM_MOE_USE_DEEP_GEMM=1` path fails as hard incompatibility on this RTX 3090:
    `FP8 MoE backend DEEPGEMM does not support ... current device cuda`.
  - It then falls back to MARLIN FP8 MoE.
- Conclusion on this stack (single 3090, vLLM nightly, this FP8 checkpoint/export):
  - `FP8 weights + vLLM nightly + single 3090` is currently a hard block due to contiguous allocation requirements in MARLIN FP8 MoE weight creation.
- Practical next move is to avoid FP8 MoE for this stack and use alternative paths (BF16 or AWQ targets) to keep lab progress moving while preserving the same vLLM OpenAI-compatible workflow.

## Option B Status (AWQ 4-bit)

- AWQ directory validated:
  - `/data/models/QuantTrio/Qwen3.5-27B-AWQ` exists and contains `config.json`.
- Attempted paths on this host image:
  - `vllm/vllm-openai:cu130-nightly-x86_64` (v0.16.1rc1.dev48+ga572baff5) is affected by a runtime regression in `RMSNormGated` and fails during AWQ init.
  - `vllm/vllm-openai:nightly` does not show this regression.
  1) `make run-qwen35-awq-boot`  
     - Starts model load and conversion to `awq_marlin`.
     - Fails during engine dummy-run path with:
       `AttributeError: 'RMSNormGated' object has no attribute 'activation'`.
  2) `make run-qwen35-awq-fast`  
     - Loads weights and reaches profile/compile stage.
     - Fails at KV-cache init with:
       `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`.
     - Log line indicates `Available KV cache memory: -1.62 GiB` despite successful weight load (`21.38 GiB`).
- With current nightly build, AWQ single-GPU on this 3090 host is **not yet serviceable** from these settings.

## Current status (2026-03-01)

- AWQ TP2 fast/boot is currently expected to be run against `vllm/vllm-openai:nightly` when this regression is present:
  - `--tensor-parallel-size 2`
  - `--language-model-only`
  - `OMP_NUM_THREADS=4`
- `make bench` now works via a dedicated benchmark helper script.
- Use `IMAGE=vllm/vllm-openai:nightly` (or your validated fixed build) when running AWQ TP2.
- Latest measured throughput (latest successful TP2 run):
  - command:
    ```sh
    python3 - <<'PY'
    import json, time, statistics, urllib.request
    url = "http://127.0.0.1:8000/v1/completions"
    payload = {
      "model": "qwen35a3b-awq",
      "prompt": "Explain tensor parallelism in language model inference with one short paragraph.",
      "temperature": 0.2,
      "top_p": 0.9,
      "max_tokens": 4,
    }
    headers = {
      "Content-Type": "application/json",
      "Authorization": "Bearer local",
      "Connection": "close",
    }
    body = json.dumps(payload).encode()
    latencies = []
    total_tokens = 0
    for _ in range(30):
      req = urllib.request.Request(url, data=body, headers=headers, method="POST")
      start = time.perf_counter()
      with urllib.request.urlopen(req, timeout=120) as resp:
          data = json.loads(resp.read().decode())
      latencies.append(time.perf_counter() - start)
      total_tokens += data.get("usage", {}).get("completion_tokens", 0)
    print(f"total_tokens={total_tokens}")
    print(f"wall={sum(latencies):.3f}s")
    print(f"tok/s={total_tokens / sum(latencies):.2f}")
    print(f"p50={statistics.median(latencies):.3f}s")
    PY
    ```
  - Result: `120 / 13.040s = 9.20 tok/s` (single request at a time, 30 runs, max_tokens=4).
  - Caveat: outputs from this run were low-quality/garbage, so this is an inferential throughput-only baseline rather than a quality benchmark.

- For the 35B AWQ run, the default target is `AWQ_VARIANT=35B` (`/data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ`).

### 35B AWQ TP2 workflow

- Start TP2 inference:
  - `make run-qwen35-awq-tp2-fast`
  - `make run-qwen35-awq-tp2-boot` (if you want non-speculative boot mode)
- Run context/throughput sweep:
  - `make bench-context-sweep BENCH_CONTEXT_PROMPT_WORDS="1024 2048 3072 4096 8192 12288" BENCH_MAX_TOKENS=256 BENCH_MODEL_NAME=qwen35a3b-awq`
- Tune memory budget per context target:
  - `make run-qwen35-awq-tp2-fast MAX_MODEL_LEN_AWQ_TP2=262144 GPU_MEM_UTIL_AWQ_TP2_FAST=0.94`
  - If startup fails at cache init, lower `MAX_MODEL_LEN_AWQ_TP2` first.

## Practical throughput tuning

- Run with higher in-flight requests once generation quality is acceptable:
  - `BENCH_REQUESTS=4`
  - `BENCH_CONCURRENCY=4`
- For chat-completions benchmarking use `make bench`:
  - `make bench BENCH_URL=http://127.0.0.1:8000/v1/chat/completions BENCH_MODEL_NAME=qwen35a3b-awq BENCH_RUNS=3`
- Completions-endpoint throughput can be measured with the custom snippet in the status section; no longer using 463 tok/s as a headline number.
- Keep `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, and KV cache settings aligned to your expected traffic profile.
- Remove tool/structured-output features when benchmarking for raw speed:
  - drop `--enable-auto-tool-choice`
  - drop `--reasoning-parser`
- For long X11 sessions, consider headless mode (`xhost -` and no active display) only if your launch script does not rely on GUI utilities; GPU is usually the main limiter.
