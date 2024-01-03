# docker build -t fschlatt/sparse-cross-encoder:0.0.1-dev -f docker/Dockerfile.dev .
FROM fschlatt/window-matmul:0.0.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y git wget curl build-essential htop

RUN conda install -y \
    pandas \
    jupyter \
    matplotlib \
    black \
    isort \
    lightning \
    transformers \
    && conda clean -y -a

RUN pip install \
    pyserini \
    ir-datasets \
    jsonargparse[signatures]>=4.18.0 \
    aquarel \
    trectools \
    && pip cache purge
