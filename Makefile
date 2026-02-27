.PHONY: up down logs status run-qwen35-boot run-qwen35-fast run-qwen35-bf16-boot run-qwen35-bf16-fast run-qwen35-awq-boot bench bench-bf16 bench-fp8

CONTAINER_NAME := vllm-qwen35
IMAGE := vllm/vllm-openai:cu130-nightly-x86_64

MODEL_PATH_FP8 := /data/models/Qwen3.5-35B-A3B-FP8
MODEL_PATH_BF16 := /data/models/Qwen/Qwen3.5-35B-A3B
MODEL_PATH_AWQ := /data/models/Qwen3.5-35B-A3B-AWQ
MODEL_PATH := $(MODEL_PATH_FP8)
CACHE_PATH := $(HOME)/.cache/vllm

GPU_IDS := 1
TP_SIZE := 1
MAX_MODEL_LEN := 262144
MAX_NUM_SEQS := 1

SERVED_MODEL_NAME_FP8 := qwen35a3b-fp8
SERVED_MODEL_NAME_BF16 := qwen35a3b-bf16
SERVED_MODEL_NAME_AWQ := qwen35a3b-awq
BENCH_MODEL_NAME ?= $(SERVED_MODEL_NAME_BF16)

GPU_MEM_UTIL_BOOT := 0.70
GPU_MEM_UTIL_FAST := 0.84
GPU_MEM_UTIL_BF16_BOOT := 0.78
GPU_MEM_UTIL_BF16_FAST := 0.84
GPU_MEM_UTIL_AWQ_BOOT := 0.90

MAX_NUM_BATCHED_TOKENS_BOOT := 1024
MAX_NUM_BATCHED_TOKENS_FAST := 4096
MAX_NUM_BATCHED_TOKENS_BF16_BOOT := 2048
MAX_NUM_BATCHED_TOKENS_BF16_FAST := 4096
MAX_NUM_BATCHED_TOKENS_AWQ_BOOT := 8192

RUNTIME_ARGS_BASE := --language-model-only --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --api-key local --served-model-name $(SERVED_MODEL_NAME_FP8)
RUNTIME_ARGS_BASE_BF16 := --language-model-only --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --api-key local --served-model-name $(SERVED_MODEL_NAME_BF16)
RUNTIME_ARGS_BASE_AWQ := --language-model-only --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --api-key local --served-model-name $(SERVED_MODEL_NAME_AWQ)
RUNTIME_ARGS_TOOLS := --enable-auto-tool-choice --tool-call-parser qwen3_coder
TOOL_CALLING ?= 0

RUNTIME_ARGS := $(RUNTIME_ARGS_BASE)
RUNTIME_ARGS_BF16 := $(RUNTIME_ARGS_BASE_BF16)
RUNTIME_ARGS_AWQ := $(RUNTIME_ARGS_BASE_AWQ)
ifeq ($(TOOL_CALLING),1)
RUNTIME_ARGS += $(RUNTIME_ARGS_TOOLS)
RUNTIME_ARGS_BF16 += $(RUNTIME_ARGS_TOOLS)
RUNTIME_ARGS_AWQ += $(RUNTIME_ARGS_TOOLS)
endif

COMMON_DOCKER_ARGS := \
  --gpus '"device=$(GPU_IDS)"' --ipc=host \
  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/compat \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p 8000:8000 \
  -v $(CACHE_PATH):/cache \
  --restart no

run-qwen35-boot:
# Known: on RTX 3090 this fp8 path often fails at init with MARLIN FP8 MoE OOM.
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
	  -v $(MODEL_PATH_FP8):/model:ro \
	  $(IMAGE) \
	  /model \
	  --host 0.0.0.0 --port 8000 \
	  --tensor-parallel-size $(TP_SIZE) \
	  --gpu-memory-utilization $(GPU_MEM_UTIL_BOOT) \
	  --max-model-len $(MAX_MODEL_LEN) \
	  --max-num-seqs $(MAX_NUM_SEQS) \
	  --max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS_BOOT) \
	  --enforce-eager \
	  $(RUNTIME_ARGS)
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

run-qwen35-fast:
# Known: on RTX 3090 this fp8 fast path is often blocked by the same init OOM behavior.
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
	  -v $(MODEL_PATH_FP8):/model:ro \
	  $(IMAGE) \
	  /model \
	  --host 0.0.0.0 --port 8000 \
	  --tensor-parallel-size $(TP_SIZE) \
	  --gpu-memory-utilization $(GPU_MEM_UTIL_FAST) \
	  --max-model-len $(MAX_MODEL_LEN) \
	  --max-num-seqs $(MAX_NUM_SEQS) \
	  --max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS_FAST) \
	  --enforce-eager \
	  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
	  $(RUNTIME_ARGS)
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

run-qwen35-bf16-boot:
	@printf 'Starting BF16 boot path on $(MODEL_PATH_BF16)\n'
	@test -d $(MODEL_PATH_BF16) || (echo "Missing BF16 model directory: $(MODEL_PATH_BF16). Download required." && exit 1)
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
	  -v $(MODEL_PATH_BF16):/model:ro \
	  $(IMAGE) \
	  /model \
	  --host 0.0.0.0 --port 8000 \
	  --tensor-parallel-size $(TP_SIZE) \
	  --gpu-memory-utilization $(GPU_MEM_UTIL_BF16_BOOT) \
	  --max-model-len $(MAX_MODEL_LEN) \
	  --max-num-seqs $(MAX_NUM_SEQS) \
	  --max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS_BF16_BOOT) \
	  --enforce-eager \
	  $(RUNTIME_ARGS_BF16)
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

run-qwen35-bf16-fast:
	@printf 'Starting BF16 fast path on $(MODEL_PATH_BF16)\n'
	@test -d $(MODEL_PATH_BF16) || (echo "Missing BF16 model directory: $(MODEL_PATH_BF16). Download required." && exit 1)
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
	  -v $(MODEL_PATH_BF16):/model:ro \
	  $(IMAGE) \
	  /model \
	  --host 0.0.0.0 --port 8000 \
	  --tensor-parallel-size $(TP_SIZE) \
	  --gpu-memory-utilization $(GPU_MEM_UTIL_BF16_FAST) \
	  --max-model-len $(MAX_MODEL_LEN) \
	  --max-num-seqs $(MAX_NUM_SEQS) \
	  --max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS_BF16_FAST) \
	  --enforce-eager \
	  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
	  $(RUNTIME_ARGS_BF16)
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

run-qwen35-awq-boot:
	@printf 'Starting AWQ boot path on $(MODEL_PATH_AWQ)\n'
	@test -d $(MODEL_PATH_AWQ) || (echo "Missing AWQ model directory: $(MODEL_PATH_AWQ). Download required." && exit 1)
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
	  -v $(MODEL_PATH_AWQ):/model:ro \
	  $(IMAGE) \
	  /model \
	  --host 0.0.0.0 --port 8000 \
	  --tensor-parallel-size $(TP_SIZE) \
	  --gpu-memory-utilization $(GPU_MEM_UTIL_AWQ_BOOT) \
	  --max-model-len $(MAX_MODEL_LEN) \
	  --max-num-seqs $(MAX_NUM_SEQS) \
	  --max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS_AWQ_BOOT) \
	  --enforce-eager \
	  $(RUNTIME_ARGS_AWQ)
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

up: run-qwen35-fast

down:
	docker rm -f $(CONTAINER_NAME)

logs:
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

status:
	docker ps -a --filter name=$(CONTAINER_NAME)

bench:
	python3 - <<-'PY'
	import json
	import time
	import urllib.request
	
	URL = "http://localhost:8000/v1/chat/completions"
	HEADERS = {
	  "Content-Type": "application/json",
	  "Authorization": "Bearer local",
	}
	MODEL = "$(BENCH_MODEL_NAME)"
	payload = {
	  "model": MODEL,
	  "messages": [{"role": "user", "content": "Explain tensor parallelism concisely, then give one example."}],
	  "temperature": 0.0,
	  "max_tokens": 512,
	  "stream": False,
	}
	
	
	def run():
	  req = urllib.request.Request(
	    URL,
	    data=json.dumps(payload).encode(),
	    headers=HEADERS,
	    method="POST",
	  )
	  t0 = time.perf_counter()
	  with urllib.request.urlopen(req, timeout=600) as response:
	    result = json.loads(response.read().decode())
	  t1 = time.perf_counter()
	  completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
	  return (t1 - t0), completion_tokens
	
	
	time_s, tokens = run()
	print(f"warmup: {time_s:.3f}s  completion={tokens}  decode={tokens/time_s:.1f} tok/s")
	for i in range(5):
	  time_s, tokens = run()
	  print(f"run{i + 1}: {time_s:.3f}s  completion={tokens}  decode={tokens/time_s:.1f} tok/s")
	PY

bench-bf16:
	$(MAKE) BENCH_MODEL_NAME=$(SERVED_MODEL_NAME_BF16) bench

bench-fp8:
	$(MAKE) BENCH_MODEL_NAME=$(SERVED_MODEL_NAME_FP8) bench
