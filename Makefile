.PHONY: up down logs status

CONTAINER_NAME := vllm-qwen35
IMAGE := vllm/vllm-openai:nightly
MODEL_PATH := /data/models/Qwen3.5-35B-A3B-FP8
CACHE_PATH := $(HOME)/.cache/vllm
GPU_IDS := 1
TENSOR_PARALLEL_SIZE := 1
GPU_MEMORY_UTILIZATION := 0.90

up:
	-@docker rm -f $(CONTAINER_NAME) >/dev/null 2>&1
	docker run -d --name $(CONTAINER_NAME) \
	  --gpus all --ipc=host \
	  -e CUDA_VISIBLE_DEVICES=$(GPU_IDS) \
	  -e LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
	  -e LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/local/cuda/compat \
	  -p 8000:8000 \
	  -v $(MODEL_PATH):/model:ro \
	  -v $(CACHE_PATH):/cache \
	  --restart no \
	  $(IMAGE) \
	  --model /model \
	  --host 0.0.0.0 --port 8000 \
	  --tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
	  --language-model-only \
	  --gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
	  --max-model-len 262144 \
	  --max-num-seqs 1 \
	  --kv-cache-dtype fp8_e4m3 \
	  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
	  --enable-chunked-prefill --max-num-batched-tokens 8192 \
	  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
	  --api-key local \
	  --trust-remote-code
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

down:
	docker rm -f $(CONTAINER_NAME)

logs:
	docker logs --follow $(CONTAINER_NAME) 2>&1 | tee out.log

status:
	docker ps -a --filter name=$(CONTAINER_NAME)
