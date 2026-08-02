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
# Pinned, not floating. C_max, S and TTFAB are measured per configuration and compared
# across configurations, so a rebuild that also moves torch makes any change in those
# numbers unattributable between the GPU and the framework. 2.6.0+cu124 is what the
# unpinned build resolved to for the deployed image (pushed 2026-07-30), so pinning it
# is a no-op today and a guarantee tomorrow.
#
# One pin serves both ml.g5 and ml.g6, but not because the wheel targets both. Verified
# by building this file and reading `torch._C._cuda_getArchFlags()`:
#   sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90     <- no sm_89
# A10G (ml.g5) is sm_86, L4 (ml.g6) is sm_89, and a cubin for X.y runs on X.z for z >= y,
# so the sm_86 kernels execute on L4. They are not sm_89-tuned, though, so some of any
# g5-vs-g6 difference is kernel selection rather than the GPU. Measure, do not assume;
# a container that cannot load kernels at all logs "no kernel image is available" from
# the warm-up in serve.py, which is the signature to check on the first g6 start.
#
# Bumping this pin invalidates every artifact's fingerprint, which is the mechanism
# working: the image digest changes, and `tts-bench` refuses to replay a measurement
# taken on the old build.
RUN pip install --no-cache-dir torch==2.6.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY serve.py .

RUN printf '#!/bin/bash\nexec python /app/serve.py\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true

EXPOSE 8080

CMD ["python", "serve.py"]
