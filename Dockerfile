FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /trt
RUN apt update && apt install -y build-essential

COPY . .

RUN pip install .

ENV TOKENIZATION_REPAIR_DOWNLOAD_DIR=/trt/download
ENV TOKENIZATION_REPAIR_CACHE_DIR=/trt/cache
WORKDIR /trt/tokenization_repair
CMD /trt/tokenization_repair/help.sh
