FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y git build-essential
RUN conda update -y conda
RUN pip install git+https://github.com/webis-de/pytorch-window-matmul.git
RUN conda install -y -c conda-forge \
    pandas \
    jupyter \
    matplotlib \
    black \
    isort \
    nmslib \
    lightning \
    wandb \
    transformers
RUN conda clean -y -a
RUN pip install \
    pyserini \
    ir-datasets \
    jsonargparse[signatures]>=4.18.0 \
    aquarel \
    trectools
RUN pip cache purge