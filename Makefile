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

.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm   $(CONTAINER_NAME) || true
