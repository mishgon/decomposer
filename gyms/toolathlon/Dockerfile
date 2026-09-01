ARG TOOLATHLON_BASE_IMAGE=docker.io/lockon0927/toolathlon-task-image:1016beta
FROM ${TOOLATHLON_BASE_IMAGE}

ARG PYTHON_VERSION=3.12
ENV SUBAGENT_VENV=/opt/subagents

# Bake the pinned benchmark sources into /workspace. The image's own
# pyproject.toml, uv.lock, and .venv are intentionally preserved so that
# `uv run` in the container reuses the image environment without re-syncing,
# mirroring how the benchmark's runtime overlay keeps the image venv intact.
COPY external/toolathlon/scripts/ /workspace/scripts/
COPY external/toolathlon/configs/ /workspace/configs/
COPY external/toolathlon/utils/ /workspace/utils/
COPY external/toolathlon/main.py /workspace/main.py
COPY external/toolathlon/tasks/ /workspace/tasks/
COPY external/toolathlon/deployment/ /workspace/deployment/
COPY external/toolathlon/global_preparation/check_installation.py /workspace/global_preparation/check_installation.py
COPY external/toolathlon/local_binary/ /workspace/local_binary/

# Separate Python environment for the LangGraph subagent server.
COPY gyms/toolathlon/subagents/requirements.txt /tmp/subagent-requirements.txt
COPY pyproject.toml README.md /tmp/decomposer/
COPY src/decomposer /tmp/decomposer/src/decomposer

RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" "${SUBAGENT_VENV}" \
    && uv pip install \
        --python "${SUBAGENT_VENV}/bin/python" \
        --requirements /tmp/subagent-requirements.txt \
    && uv pip install \
        --python "${SUBAGENT_VENV}/bin/python" \
        --no-deps \
        /tmp/decomposer

COPY gyms/toolathlon /opt/decomposer/gyms/toolathlon

WORKDIR /workspace
