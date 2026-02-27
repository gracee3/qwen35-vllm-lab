.PHONY: up down logs status run-qwen35-boot run-qwen35-fast

CONTAINER_NAME := vllm-qwen35
IMAGE := vllm/vllm-openai:nightly
MODEL_PATH := /data/models/Qwen3.5-35B-A3B-FP8
CACHE_PATH := $(HOME)/.cache/vllm
GPU_IDS := 1
TP_SIZE := 1
MAX_MODEL_LEN := 262144
MAX_NUM_SEQS := 1
GPU_MEM_UTIL_BOOT := 0.82
GPU_MEM_UTIL_FAST := 0.84
MAX_NUM_BATCHED_TOKENS_BOOT := 2048
MAX_NUM_BATCHED_TOKENS_FAST := 4096
RUNTIME_ARGS := --language-model-only --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --enable-auto-tool-choice --tool-call-parser qwen3_coder --api-key local --trust-remote-code

COMMON_DOCKER_ARGS := \
	  --gpus all --ipc=host \
	  -e CUDA_VISIBLE_DEVICES=$(GPU_IDS) \
	  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
	  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/compat \
	  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
	  -p 8000:8000 \
	  -v $(MODEL_PATH):/model:ro \
	  -v $(CACHE_PATH):/cache \
	  --restart no

run-qwen35-boot:
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  $(COMMON_DOCKER_ARGS) \
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
