# Qwen3.5 vLLM launcher
# Single-file, minimal setup for 9B, 27B, and 4B local model directories.

.PHONY: setup build run-llm \
	run-qwen35-9b run-qwen35-27b run-qwen35-4b \
	stop stop-llm stop-qwen35-9b stop-qwen35-27b stop-qwen35-4b \
	logs logs-llm logs-qwen35-9b logs-qwen35-27b logs-qwen35-4b

# Images
IMAGE := vllm/vllm-openai:v0.18.0

# Paths
CACHE_PATH := $(HOME)/.cache/vllm
MODEL_PATH ?= /data/models/Qwen/Qwen3.5-9B

# Docker
DOCKER_RESTART_POLICY ?= no
DOCKER_PULL_POLICY ?= never

# Networking
PORT ?= 8000
API_KEY ?= local

# GPU pinning
# GPU=0, GPU=1, GPU=0,1, or GPU=all
GPU ?= 1
ifeq ($(GPU),all)
  GPU_FLAG := --gpus all
else
  GPU_FLAG := --gpus '"device=$(GPU)"'
endif

# vLLM defaults
CONTAINER_NAME ?= vllm-qwen35-9b
SERVED_MODEL_NAME ?= qwen35-9b
TP_SIZE ?= 1
GPU_MEM_UTIL ?= 0.92
MAX_MODEL_LEN ?= 16384
MAX_NUM_SEQS ?= 4
MAX_NUM_BATCHED_TOKENS ?= 4096
TOOL_CALL_PARSER ?= qwen3_xml
CPU_OFFLOAD_GB ?= 0

setup:
	mkdir -p $(CACHE_PATH)

build:
ifeq ($(DOCKER_PULL_POLICY),never)
	@echo "Skipping docker pull because DOCKER_PULL_POLICY=never"
else
	docker pull $(IMAGE)
endif

run-llm: stop-llm build setup
	@test -d $(MODEL_PATH) || (echo "Missing model directory: $(MODEL_PATH)" && exit 1)
	docker run -d \
		--name $(CONTAINER_NAME) \
		$(GPU_FLAG) \
		--ipc=host \
		-p $(PORT):8000 \
		-v $(MODEL_PATH):/model:ro \
		-v $(CACHE_PATH):/root/.cache \
		--restart $(DOCKER_RESTART_POLICY) \
		--pull $(DOCKER_PULL_POLICY) \
		$(IMAGE) \
		/model \
		--host 0.0.0.0 \
		--port 8000 \
		--api-key $(API_KEY) \
		--gpu-memory-utilization $(GPU_MEM_UTIL) \
		--tensor-parallel-size $(TP_SIZE) \
		--max-model-len $(MAX_MODEL_LEN) \
		--max-num-batched-tokens $(MAX_NUM_BATCHED_TOKENS) \
		--max-num-seqs $(MAX_NUM_SEQS) \
		--cpu-offload-gb $(CPU_OFFLOAD_GB) \
		--language-model-only \
		--served-model-name $(SERVED_MODEL_NAME) \
		--reasoning-parser qwen3 \
		--enable-auto-tool-choice \
		--tool-call-parser $(TOOL_CALL_PARSER) \
		$(EXTRA_ARGS)
	docker logs -f $(CONTAINER_NAME)

run-qwen35-9b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) run-llm \
		CONTAINER_NAME=vllm-qwen35-9b \
		MODEL_PATH=/data/models/Qwen/Qwen3.5-9B \
		SERVED_MODEL_NAME=qwen35-9b \
		GPU=1 \
		TP_SIZE=1 \
		GPU_MEM_UTIL=0.92 \
		MAX_MODEL_LEN=16384 \
		MAX_NUM_SEQS=4 \
		MAX_NUM_BATCHED_TOKENS=4096

run-qwen35-27b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) run-llm \
		CONTAINER_NAME=vllm-qwen35-27b \
		MODEL_PATH=/data/models/Qwen/Qwen3.5-27B \
		SERVED_MODEL_NAME=qwen35-27b \
		GPU=0,1 \
		TP_SIZE=2 \
		GPU_MEM_UTIL=0.85 \
		MAX_MODEL_LEN=8192 \
		MAX_NUM_SEQS=1 \
		MAX_NUM_BATCHED_TOKENS=2048 \
		CPU_OFFLOAD_GB=8

run-qwen35-4b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) run-llm \
		CONTAINER_NAME=vllm-qwen35-4b \
		MODEL_PATH=/data/models/Qwen/Qwen3.5-4B \
		SERVED_MODEL_NAME=qwen35-4b \
		GPU=1 \
		TP_SIZE=1 \
		GPU_MEM_UTIL=0.92 \
		MAX_MODEL_LEN=16384 \
		MAX_NUM_SEQS=4 \
		MAX_NUM_BATCHED_TOKENS=4096

up: run-qwen35-9b

stop: stop-qwen35-9b

stop-llm:
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm $(CONTAINER_NAME) 2>/dev/null || true

stop-qwen35-9b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) stop-llm CONTAINER_NAME=vllm-qwen35-9b

stop-qwen35-27b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) stop-llm CONTAINER_NAME=vllm-qwen35-27b

stop-qwen35-4b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) stop-llm CONTAINER_NAME=vllm-qwen35-4b

logs: logs-qwen35-9b

logs-llm:
	docker logs -f $(CONTAINER_NAME)

logs-qwen35-9b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) logs-llm CONTAINER_NAME=vllm-qwen35-9b

logs-qwen35-27b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) logs-llm CONTAINER_NAME=vllm-qwen35-27b

logs-qwen35-4b:
	$(MAKE) -f $(lastword $(MAKEFILE_LIST)) logs-llm CONTAINER_NAME=vllm-qwen35-4b
