NAME := docker.io/jozoppi/inference-template:20250622232501
format:
	uv run ruff format model_inference tests models
lint:
	uv run ruff check model_inference tests models	
build-docker:
	docker buildx build . -t $(NAME)
