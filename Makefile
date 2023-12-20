SHELL=/bin/bash

.DEFAULT_GOAL=help

.PHONY: data train

dep: ## install python dependencies
	pip install -r requirements.txt
	pip install -r train/train_requirements.txt

data: ## get raw data from physionet
	wget -r -N -c -np --directory-prefix=data https://physionet.org/files/sleep-accel/1.0.0/

train: ## run model training with evaluation
	PYTHONPATH=. python train/train_engineered.py

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
