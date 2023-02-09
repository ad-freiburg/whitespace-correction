FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /wsc

COPY . .

RUN pip install .

ENV WHITESPACE_CORRECTION_DOWNLOAD_DIR=/wsc/download
ENV WHITESPACE_CORRECTION_CACHE_DIR=/wsc/cache
ENV PYTHONWARNINGS="ignore"

ENTRYPOINT ["/opt/conda/bin/wsc"]
