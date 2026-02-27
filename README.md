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
  - `/data/models/QuantTrio/Qwen3.5-35B-A3B-AWQ` exists and contains `config.json`.
- Attempted path on this host image (`vllm/vllm-openai:cu130-nightly-x86_64`, v0.16.1rc1.dev48+ga572baff5):
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
