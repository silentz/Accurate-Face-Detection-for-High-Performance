SHELL:=bash
.PHONY: unittest


unittest:
	pytest tests/


train:
	PYTHONPATH=. python cmd/train.py


debug:
	PYTHONPATH=. python3 cmd/run.py --checkpoint weights/ainnoface.pth --image data/WIDER_val/images/0--Parade/0_Parade_Parade_0_239.jpg
