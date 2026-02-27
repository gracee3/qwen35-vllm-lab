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
