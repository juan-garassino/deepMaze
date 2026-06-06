# deepMaze — local + RunPod targets
#
# Usage:
#   make test                          Run the pytest suite (CPU, fast)
#   make local                         Run the nano-local training (CPU, ~2 min)
#   make build                         Build the RunPod Docker image
#   make push REGISTRY=your-dh         Push to Docker Hub (default: garassinoj)
#   make run API_KEY=                  Run RunPod image locally (needs nvidia-docker)
#   make logs                          Tail the running container
#   make stop                          Stop + rm the container

IMAGE_NAME     ?= deepmaze-train
REGISTRY       ?= garassinoj
CONTAINER_NAME ?= deepmaze
DOCKERFILE     ?= runpod/Dockerfile

# ---------------------------------------------------------------------------
# Tests + local training
# ---------------------------------------------------------------------------

.PHONY: test
test:
	python -m pytest tests/ -q

.PHONY: local
local:
	OUTPUT_BASE=$(PWD)/local_runs \
	AGENTS_TO_RUN=drqn \
	CURRICULUM="8,8,1,200,100" \
	EXPLORATION_DECAY=0.99 \
	BUFFER_CAPACITY=2000 \
	PARTIAL=2 \
	python scripts/train_runpod.py

# ---------------------------------------------------------------------------
# Docker — build + push (for RunPod)
# ---------------------------------------------------------------------------

.PHONY: build
build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

.PHONY: push
push: build
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(IMAGE_NAME)
	docker push $(REGISTRY)/$(IMAGE_NAME)
	@echo ""
	@echo "=== Pushed $(REGISTRY)/$(IMAGE_NAME) ==="
	@echo ""
	@echo "RunPod setup:"
	@echo "  1. Create GPU Pod (A100/H100 ideal; T4 enough for DRQN)"
	@echo "  2. Container image: $(REGISTRY)/$(IMAGE_NAME)"
	@echo "  3. Volume: mount one at /workspace (training outputs go there)"
	@echo "  4. Env vars (optional): AGENTS_TO_RUN, CURRICULUM, RUN_TAG, ..."
	@echo "     See scripts/train_runpod.py docstring for the full list."
	@echo "  5. Start the pod — training begins automatically."

# ---------------------------------------------------------------------------
# Docker — run locally
# ---------------------------------------------------------------------------

.PHONY: run
run:
	docker run -d \
		--gpus all \
		--name $(CONTAINER_NAME) \
		-v $(PWD)/local_runs:/workspace \
		$(IMAGE_NAME)
	@echo ""
	@echo "=== Container started: $(CONTAINER_NAME) ==="
	@echo "    logs:  make logs"
	@echo "    stop:  make stop"

# ---------------------------------------------------------------------------
# Self-improve: train then hand control to Claude Code (same pattern as
# 005-products/020-autoresearch). Requires ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------

.PHONY: improve
improve:
ifndef API_KEY
	$(error API_KEY is required. Usage: make improve API_KEY=sk-ant-...)
endif
	docker run -d \
		--gpus all \
		--name $(CONTAINER_NAME)-improve \
		-e CLAUDE_SELF_IMPROVE=true \
		-e ANTHROPIC_API_KEY=$(API_KEY) \
		-e MAX_IMPROVE_ITERS=$${MAX_IMPROVE_ITERS:-5} \
		-e MAX_IMPROVE_HOURS=$${MAX_IMPROVE_HOURS:-4} \
		-v $(PWD)/local_runs:/workspace \
		$(IMAGE_NAME)
	@echo ""
	@echo "=== Self-improve container started: $(CONTAINER_NAME)-improve ==="
	@echo "    logs:  docker logs -f $(CONTAINER_NAME)-improve"
	@echo "    claude.log inside container: docker exec $(CONTAINER_NAME)-improve cat /app/claude.log"
	@echo "    stop:  docker stop $(CONTAINER_NAME)-improve && docker rm $(CONTAINER_NAME)-improve"

.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm   $(CONTAINER_NAME) || true
