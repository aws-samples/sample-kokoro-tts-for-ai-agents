FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libsndfile1 espeak-ng git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY serve.py .

RUN printf '#!/bin/bash\nexec python /app/serve.py\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

EXPOSE 8080

CMD ["python", "serve.py"]
