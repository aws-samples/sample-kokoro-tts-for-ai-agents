FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

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
# numbers unattributable between the GPU and the framework.
#
# +cu126, not +cu124: 2.6.0+cu124's own pinned dependency nvidia-cudnn-cu12==9.1.0.70
# was removed from the index on or before 2026-08-13, breaking every fresh build of
# this exact pin with no change on this repo's side -- pip's resolver will not
# substitute a newer cudnn build while installing torch normally, since torch's own
# metadata hard-pins the exact version (confirmed: forcing a substitute produces
# ResolutionImpossible). +cu124 had no newer release on its own index to move to
# instead, so this moved to +cu126 -- same 2.6.0 torch release, linked against a
# different CUDA toolkit version, whose own pinned deps (nvidia-cudnn-cu12==9.5.1.17
# et al.) still resolve as of this change. The base image above is CUDA 12.6 to match.
#
# One pin serving both ml.g5 (A10G, sm_86) and ml.g6 (L4, sm_89) was verified for the
# +cu124 build by reading `torch._C._cuda_getArchFlags()` (sm_50/60/70/75/80/86/90, no
# sm_89, forward-compatible within major arch). That check has NOT been re-run against
# this +cu126 build -- it needs a GPU to execute, which this dev environment does not
# have. The live deployment this pin actually serves today is ml.g5.xlarge (sm_86)
# only, so this pin is proven correct for that by the redeploy + live verification this
# change went through; sm_89/ml.g6 compatibility is carried forward as an assumption
# from the +cu124 build, not re-verified. A container that cannot load kernels at all
# logs "no kernel image is available" from the warm-up in serve.py -- the signature to
# check first if a future ml.g6 deploy behaves oddly.
#
# Bumping this pin invalidates every artifact's fingerprint, which is the mechanism
# working: the image digest changes, and `tts-bench` refuses to replay a measurement
# taken on the old build.
RUN pip install --no-cache-dir torch==2.6.0+cu126 \
        --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY serve.py .

RUN printf '#!/bin/bash\nexec python /app/serve.py\n' > /usr/local/bin/serve \
    && chmod +x /usr/local/bin/serve

LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true

EXPOSE 8080

CMD ["python", "serve.py"]
