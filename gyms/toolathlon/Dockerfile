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
COPY gyms/toolathlon/patches/12306-mcp-cookie.patch /tmp/12306-mcp-cookie.patch

# 12306 retired /otn/lcQuery/init, and its replacement now indents the
# lc_search_url declaration with a tab.  Both changes make the benchmark's
# pinned 12306-mcp@0.3.5 abort before opening its stdio transport. Keep that
# package and change only how it discovers the unchanged query URL.
RUN grep -q "https://kyfw.12306.cn/otn/lcQuery/init" \
        /workspace/node_modules/12306-mcp/build/index.js \
    && grep -Fq "html.match(/ var lc_search_url" \
        /workspace/node_modules/12306-mcp/build/index.js \
    && sed -i \
        's#https://kyfw.12306.cn/otn/lcQuery/init#https://kyfw.12306.cn/otn/leftTicket/init#g' \
        /workspace/node_modules/12306-mcp/build/index.js \
    && sed -i \
        's#html.match(/ var lc_search_url#html.match(/\\s*var lc_search_url#' \
        /workspace/node_modules/12306-mcp/build/index.js \
    && ! grep -q "https://kyfw.12306.cn/otn/lcQuery/init" \
        /workspace/node_modules/12306-mcp/build/index.js \
    && grep -Fq "html.match(/\\s*var lc_search_url" \
        /workspace/node_modules/12306-mcp/build/index.js \
    && patch --batch /workspace/node_modules/12306-mcp/build/index.js \
        < /tmp/12306-mcp-cookie.patch

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

COPY gyms/toolathlon/podman-wrapper.sh /opt/decomposer/bin/podman
RUN chmod 0755 /opt/decomposer/bin/podman
ENV PATH=/opt/decomposer/bin:${PATH}
COPY gyms/toolathlon /opt/decomposer/gyms/toolathlon

WORKDIR /workspace
