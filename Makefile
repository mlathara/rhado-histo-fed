.PHONY: format lint
default: format

format:
	black --line-length 100 sarcoma-histo-fed/custom
	isort --profile black sarcoma-histo-fed/custom

lint:
	bandit -r sarcoma-histo-fed/custom
	black --check --line-length 100 sarcoma-histo-fed/custom
	isort --profile black -c sarcoma-histo-fed/custom