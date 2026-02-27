.PHONY: up down logs status run-qwen35-boot run-qwen35-fast bench

CONTAINER_NAME := vllm-qwen35
IMAGE := vllm/vllm-openai:cu130-nightly-x86_64
MODEL_PATH := /data/models/Qwen3.5-35B-A3B-FP8
CACHE_PATH := $(HOME)/.cache/vllm
GPU_IDS := 1
TP_SIZE := 1
MAX_MODEL_LEN := 262144
MAX_NUM_SEQS := 1
SERVED_MODEL_NAME := qwen35a3b-fp8
GPU_MEM_UTIL_BOOT := 0.70
GPU_MEM_UTIL_FAST := 0.84
MAX_NUM_BATCHED_TOKENS_BOOT := 1024
MAX_NUM_BATCHED_TOKENS_FAST := 4096
RUNTIME_ARGS_BASE := --language-model-only --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --api-key local --served-model-name $(SERVED_MODEL_NAME)
RUNTIME_ARGS_TOOLS := --enable-auto-tool-choice --tool-call-parser qwen3_coder
TOOL_CALLING ?= 0
RUNTIME_ARGS := $(RUNTIME_ARGS_BASE)
ifeq ($(TOOL_CALLING),1)
RUNTIME_ARGS += $(RUNTIME_ARGS_TOOLS)
endif

COMMON_DOCKER_ARGS := \
	  --gpus '"device=$(GPU_IDS)"' --ipc=host \
	  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
	  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/compat \
	  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
	  -p 8000:8000 \
	  -v $(MODEL_PATH):/model:ro \
	  -v $(CACHE_PATH):/cache \
	  --restart no

BOOT_MOE_ARGS := \
	  -e VLLM_USE_FLASHINFER_MOE_FP16=1

run-qwen35-boot:
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
	  $(BOOT_MOE_ARGS) \
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
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
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
	MODEL = "qwen35a3b-fp8"
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
