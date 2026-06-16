FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VOICES_DIR=/app/voices
ENV DEFAULT_VOICE=female_shadowheart4
ENV GPU_MEMORY_UTILIZATION=0.7
ENV MAX_MODEL_LEN=1000
ENV MODEL_DIR=/app/model

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    libsndfile1 git curl unzip gcc \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone --depth 1 https://github.com/randombk/chatterbox-vllm.git /tmp/chatterbox-vllm \
    && pip install --no-cache-dir --no-deps /tmp/chatterbox-vllm \
    && cp -r /tmp/chatterbox-vllm/t3-model /app/t3-model \
    && rm -rf /tmp/chatterbox-vllm

COPY vllm_plugin.py .
COPY setup.cfg .
COPY setup.py .
RUN pip install --no-cache-dir -e .

COPY voices/ /app/voices/
COPY streaming_proxy.py .
COPY entrypoint.sh .
RUN chmod +x /app/entrypoint.sh

RUN printf '#!/bin/bash\nexec /app/entrypoint.sh\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

EXPOSE 8080

CMD ["/app/entrypoint.sh"]
