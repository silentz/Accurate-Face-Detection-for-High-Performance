SHELL:=bash
.PHONY: unittest


unittest:
	pytest tests/


train:
	PYTHONPATH=. python cmd/train.py

