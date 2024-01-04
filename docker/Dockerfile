FROM fschlatt/sparse-cross-encoder:0.0.1-dev

ARG JUPYTER_NOTEBOOK=run-sparse-cross-encoder-4-4096.ipynb

COPY . /workspaces/ecir24-sparse-cross-encoder

RUN wget https://raw.githubusercontent.com/tira-io/ir-experiment-platform/main/tira-ir-starters/pyterrier/run-pyterrier-notebook.py -O /workspace/run-pyterrier-notebook.py \
    && chmod +x /workspace/run-pyterrier-notebook.py \
    && cp /workspaces/ecir24-sparse-cross-encoder/notebooks/${JUPYTER_NOTEBOOK} /workspaces/ecir24-sparse-cross-encoder/notebooks/notebook.ipynb \
    && pip3 install runnb \
    && jupyter trust /workspaces/ecir24-sparse-cross-encoder/notebooks/*.ipynb

ENTRYPOINT [ "/workspace/run-pyterrier-notebook.py", "--input", "$inputDataset", "--output", "$outputDir", "--notebook", "/workspaces/ecir24-sparse-cross-encoder/notebooks/notebook.ipynb" ]
