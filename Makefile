# deepMaze — local + Docker + RunPod targets
#
# Local + Docker:
#   make test                          Run the pytest suite (CPU, fast)
#   make local                         Nano local training (CPU, ~2 min)
#   make build                         Build the training Docker image
#   make run                           Run the image locally with --gpus all
#   make improve API_KEY=sk-ant-...    Train + Claude self-improve loop locally
#   make logs / make stop              Tail / stop the local container
#
# RunPod (push image → ghcr.io → create pod):
#   make ghcr-login                    docker login ghcr.io (paste GitHub PAT)
#   make push                          build + push image to ghcr.io/$(GH_USERNAME)
#   make runpod                        runpodctl pod create — training only
#   make runpod-improve API_KEY=...    runpodctl pod create — train + self-improve
#   make runpod-list                   list all pods + ids
#   make runpod-get   POD_ID=...       pod details (image, status, IP, ports)
#   make runpod-stop  POD_ID=...       stop a running pod
#   make runpod-delete POD_ID=...      delete a pod
# (Pod logs: ssh into the pod via `runpodctl ssh connect POD_ID`, then
#  `tail -f /app/claude.log` or `/workspace/improve_log.tsv`.)

IMAGE_NAME     ?= deepmaze-train
GH_USERNAME    ?= juan-garassino
REGISTRY       ?= ghcr.io/$(GH_USERNAME)
CONTAINER_NAME ?= deepmaze
DOCKERFILE     ?= runpod/Dockerfile

# RunPod defaults — override on the command line if needed.
GPU_ID            ?= NVIDIA GeForce RTX 4090
POD_NAME          ?= deepmaze-train
CONTAINER_DISK_GB ?= 20
VOLUME_GB         ?= 50

# Self-improve caps (passed through to the container as env vars).
MAX_IMPROVE_ITERS ?= 5
MAX_IMPROVE_HOURS ?= 4

# ---------------------------------------------------------------------------
# Tests + local training (no Docker)
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
# Docker — build + push to GHCR
# ---------------------------------------------------------------------------

.PHONY: build
build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

.PHONY: ghcr-login
ghcr-login:
	@echo "Paste a GitHub Personal Access Token with write:packages scope."
	@echo "Create one at: https://github.com/settings/tokens"
	@docker login ghcr.io -u $(GH_USERNAME)

.PHONY: push
push: build
	docker tag $(IMAGE_NAME) $(REGISTRY)/$(IMAGE_NAME):latest
	docker push $(REGISTRY)/$(IMAGE_NAME):latest
	@echo ""
	@echo "=== Pushed $(REGISTRY)/$(IMAGE_NAME):latest ==="
	@echo ""
	@echo "If this is the first push, the package starts PRIVATE."
	@echo "Make it public so RunPod can pull without registry-auth:"
	@echo "  https://github.com/users/$(GH_USERNAME)/packages/container/$(IMAGE_NAME)/settings"
	@echo "  → Change visibility → Public"
	@echo ""
	@echo "Then: make runpod                  (train-only)"
	@echo "   or: make runpod-improve API_KEY=sk-ant-...  (self-improve)"

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
		-e MAX_IMPROVE_ITERS=$(MAX_IMPROVE_ITERS) \
		-e MAX_IMPROVE_HOURS=$(MAX_IMPROVE_HOURS) \
		-v $(PWD)/local_runs:/workspace \
		$(IMAGE_NAME)
	@echo ""
	@echo "=== Self-improve container started: $(CONTAINER_NAME)-improve ==="
	@echo "    logs:  docker logs -f $(CONTAINER_NAME)-improve"
	@echo "    claude.log: docker exec $(CONTAINER_NAME)-improve cat /app/claude.log"
	@echo "    stop:  docker stop $(CONTAINER_NAME)-improve && docker rm $(CONTAINER_NAME)-improve"

.PHONY: logs
logs:
	docker logs -f $(CONTAINER_NAME)

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm   $(CONTAINER_NAME) || true

# ---------------------------------------------------------------------------
# RunPod — create / list / logs / stop pods via runpodctl
# Requires: brew install runpod/runpodctl/runpodctl  +  export RUNPOD_API_KEY=...
# Requires: image already pushed via `make push` (and made public on GHCR).
# ---------------------------------------------------------------------------

.PHONY: runpod
runpod:
	runpodctl pod create \
		--name $(POD_NAME) \
		--image $(REGISTRY)/$(IMAGE_NAME):latest \
		--gpu-id "$(GPU_ID)" \
		--container-disk-in-gb $(CONTAINER_DISK_GB) \
		--volume-in-gb $(VOLUME_GB) \
		--volume-mount-path /workspace

.PHONY: runpod-improve
runpod-improve:
ifndef API_KEY
	$(error API_KEY is required. Usage: make runpod-improve API_KEY=sk-ant-...)
endif
	runpodctl pod create \
		--name $(POD_NAME)-improve \
		--image $(REGISTRY)/$(IMAGE_NAME):latest \
		--gpu-id "$(GPU_ID)" \
		--container-disk-in-gb $(CONTAINER_DISK_GB) \
		--volume-in-gb $(VOLUME_GB) \
		--volume-mount-path /workspace \
		--env '{"CLAUDE_SELF_IMPROVE":"true","ANTHROPIC_API_KEY":"$(API_KEY)","MAX_IMPROVE_ITERS":"$(MAX_IMPROVE_ITERS)","MAX_IMPROVE_HOURS":"$(MAX_IMPROVE_HOURS)"}'

.PHONY: runpod-list
runpod-list:
	runpodctl pod list

# Pod details (image, GPU, status, IP, ports). Override POD_ID:
#   make runpod-get POD_ID=abc123
.PHONY: runpod-get
runpod-get:
ifndef POD_ID
	$(error POD_ID is required. Usage: make runpod-get POD_ID=... (find via `make runpod-list`))
endif
	runpodctl pod get $(POD_ID)

# runpodctl has no built-in `pod logs` — tail via SSH instead:
#   runpodctl ssh connect $(POD_ID)
# then on the pod: tail -f /app/claude.log  (or /workspace/improve_log.tsv)

.PHONY: runpod-stop
runpod-stop:
ifndef POD_ID
	$(error POD_ID is required. Usage: make runpod-stop POD_ID=... (find via `make runpod-list`))
endif
	runpodctl pod stop $(POD_ID)

.PHONY: runpod-delete
runpod-delete:
ifndef POD_ID
	$(error POD_ID is required. Usage: make runpod-delete POD_ID=...)
endif
	runpodctl pod delete $(POD_ID)
