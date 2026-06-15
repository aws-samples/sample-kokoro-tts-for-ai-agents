FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    curl unzip libsndfile1 gcc \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    vllm==0.8.5.post1 \
    "transformers>=4.48,<5.0" \
    starlette uvicorn httpx websockets snac numpy soundfile

RUN curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip \
    && unzip -q /tmp/aws.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/aws.zip

ARG ADOT_VERSION=v0.41.1
RUN curl -sL -o /tmp/adot.deb \
    https://aws-otel-collector.s3.amazonaws.com/ubuntu/amd64/${ADOT_VERSION}/aws-otel-collector.deb \
    && dpkg -i /tmp/adot.deb && rm /tmp/adot.deb

COPY adot-config.yaml /opt/aws/aws-otel-collector/etc/config.yaml
COPY streaming_proxy.py /opt/streaming_proxy.py
COPY tts_orpheus/ /opt/tts_orpheus/
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENV PYTHONPATH="/opt:${PYTHONPATH}"
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
ENTRYPOINT ["/opt/entrypoint.sh"]
