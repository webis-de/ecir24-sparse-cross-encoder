FROM fschlatt/sparse-cross-encoder:0.0.1-dev

ARG JUPYTER_NOTEBOOK=run-sparse-cross-encoder-4-4096.ipynb

COPY . /workspace

RUN wget https://raw.githubusercontent.com/tira-io/ir-experiment-platform/main/tira-ir-starters/pyterrier/run-pyterrier-notebook.py -O /workspace/run-pyterrier-notebook.py \
    && chmod +x /workspace/run-pyterrier-notebook.py \
    && cp /workspace/notebook/${JUPYTER_NOTEBOOK} /workspace/notebook.ipynb \
    && pip3 install runnb \
    && jupyter trust /workspace/*.ipynb

ENTRYPOINT [ "/workspace/run-pyterrier-notebook.py", "--input", "$inputDataset", "--output", "$outputDir", "--notebook", "/workspace/notebook.ipynb" ]