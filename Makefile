.DEFAULT_GOAL := help

.venv:
	python3 -m venv .venv


.PHONY: venv
venv: .venv ## creates virtualenv .venv and updates package managers
	.venv/bin/python3 -m pip install --upgrade \
		pip \
		flit
	@echo "type 'source .venv/bin/activate' to activate your venv"
	

.PHONY: install-dev
install-dev: .venv ## installs for development (symlink, tools and testing)
	# flit install --symlink
	pip install -e ".[test,dev]"


.PHONY: info
info: ## displays info
	pip list


.PHONY: clean
clean: .check-clean ## cleans unversioned/ignored files
	@git clean -dfx -e .vscode

.check-clean:
	@git clean -ndfx -e .vscode
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

.PHONY: help
help: ## this colorful help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)