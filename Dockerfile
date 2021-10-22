FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONBREAKPOINT=ipdb.set_trace
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# copy kaggle auth
COPY kaggle.json /root/.kaggle/kaggle.json
COPY requirements.txt /workspace/requirements.txt

# install extra python packages
RUN pip install -r requirements.txt
RUN pip install dvc --ignore-installed ruamel-yaml
RUN CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall pillow-simd

# install personal library
COPY src /src
RUN cd /src && pip install -e .

# verify pillow-simd + libjpegturbo installation
RUN python -c "from PIL import features; print(features.check_feature('libjpeg_turbo'))"
