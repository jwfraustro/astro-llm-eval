# Agent sandbox: the environment the model under evaluation works in.
#
# Pinned on purpose. astroquery's Gaia module and pyvo have both changed
# behaviour across releases, and "the library changed" must never be confused
# with "the model got worse". Bump pins deliberately, in their own commit.
#
# The container runs as root with outbound network enabled: the task *is* to
# reach remote TAP services, and pip installs are part of what we measure.
# No credentials are mounted; the verifier never runs in here.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    "numpy==2.5.4" \
    "pandas==3.1.1" \
    "requests==2.34.2" \
    "astropy==8.0.2" \
    "astroquery==0.4.11" \
    "pyvo==1.10.0"

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
