.PHONY: install train test clean

install:
	uv sync
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

train:
	python -m src.deep_learning.training

test:
	python -m pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete