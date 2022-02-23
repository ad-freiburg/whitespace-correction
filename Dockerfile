FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /trt
ENV PYTHONPATH "$PYTHONPATH:/trt"

COPY . .

# general requirements
RUN pip install -r requirements.txt
# tokenization repair specific requirements
RUN pip install -r tokenization_repair/requirements.txt

WORKDIR /trt/tokenization_repair

CMD /trt/tokenization_repair/welcome.sh
