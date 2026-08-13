# Build context is the repo root (see stacks/runtime.py's DockerImageAsset) --
# the COPY packages/tts-client/ line below needs that directory available.
#
# 3.12, not 3.11 (this repo's own floor in pyproject.toml's requires-python):
# tts-client's own dependency aws-sdk-sagemaker-runtime-http2 only publishes
# wheels for Python >=3.12 -- confirmed by a real `pip install` failure under
# 3.11-slim, not assumed. The dev workspace's own .venv already runs 3.12.9.
#
# Pin this to a specific sha256 digest before production use, resolved via
# `docker pull python:3.12-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim`
# -- not fabricated here, since an unverified digest fails every build outright.
FROM python:3.12-slim

WORKDIR /app

COPY packages/agent-infra/container/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# tts-client is a normal pyproject.toml package (hatchling build backend) but
# isn't on PyPI -- pip builds and installs it directly from source, and its
# own dependencies (boto3, pydantic, aws-sdk-sagemaker-runtime-http2) resolve
# automatically from its pyproject.toml.
COPY packages/tts-client/ ./tts-client/
RUN pip install --no-cache-dir ./tts-client

COPY packages/agent-infra/container/*.py .

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')"]

EXPOSE 8080
CMD ["python", "server.py"]
