.DEFAULT_GOAL := help

PYTHON_INTERPRETER = python3

.PHONY: help
help:  ## Prints this help message
ifeq ($(shell uname -s),Darwin)
	@gawk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9/.%]+:.*?##/ { cmd = "echo \"" $$2 "\""; cmd | getline value; close(cmd); printf " \033[36m%-15s\033[0m %s\n", $$1, value } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
else
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9/.%]+:.*?##/ { cmd = "echo \"" $$2 "\""; cmd | getline value; close(cmd); printf " \033[36m%-15s\033[0m %s\n", $$1, value } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
endif
##@ Environment
.PHONY: .venv
.venv: ## Create a Python virtual environment
	$(PYTHON_INTERPRETER) -m .venv .venv

.PHONY: clean
clean: ## Remove Python file artifacts and .venv
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	rm -rf .venv

##@ Dependencies for the .venv
.PHONY: requirements_dev
requirements_dev: requirements requirements_dev.txt ## Installs dependencies from both requirements(+ _dev).txt and opens the package in developement mode
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install -r requirements_dev.txt

.PHONY: requirements
requirements: requirements.txt ## Installs dependencies from requirements.txt
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install -r requirements.txt
