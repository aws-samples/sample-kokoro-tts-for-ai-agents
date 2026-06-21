FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_DIR=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY serve.py .

RUN printf '#!/bin/bash\nexec python /app/serve.py\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

EXPOSE 8080

CMD ["python", "serve.py"]
