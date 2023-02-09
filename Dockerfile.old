FROM nvcr.io/nvidia/pytorch:22.05-py3

WORKDIR /wsc
RUN apt update && apt install -y build-essential

COPY . .

RUN pip install .

ENV WHITESPACE_CORRECTION_DOWNLOAD_DIR=/wsc/download
ENV WHITESPACE_CORRECTION_CACHE_DIR=/wsc/cache

ENTRYPOINT "wsc"
