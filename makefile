format:
	isort . && \
	black -l 79 .

install:
	cd src && \
	pip install -e . && \
	cd ..

build:
	docker build -t kaggle .

start:
	nvidia-docker run -d --name paw --ipc=host --gpus all -p 5000:5000 -p 8888:8888 --rm -v "/home/gianluca/git/kaggle/paw:/workspace" -v "/data:/data" -t kaggle

attach:
	docker exec -it paw /bin/bash

stop:
	docker kill paw

upload_lib:
	kaggle datasets version --dir-mode zip -p src/ -m "New version"

upload_ckpts:
	kaggle datasets version --dir-mode zip -p ckpts/ -m "New version"

jupyter:
	jupyter lab --ip 0.0.0.0 --port 8888
