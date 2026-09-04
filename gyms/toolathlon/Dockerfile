ARG TOOLATHLON_BASE_IMAGE=docker.io/lockon0927/toolathlon-task-image:1016beta
FROM ${TOOLATHLON_BASE_IMAGE}

ARG PYTHON_VERSION=3.12
ENV SUBAGENT_VENV=/opt/subagents

# Bitnami's legacy chart endpoint now redirects to a Cloudflare-protected URL
# that rejects non-browser clients. Pull the same immutable chart version from
# Bitnami's OCI registry once at build time so K8s preprocessing is offline.
RUN mkdir -p /opt/decomposer/vendor \
    && helm pull oci://registry-1.docker.io/bitnamicharts/redis \
        --version 19.0.0 \
        --destination /opt/decomposer/vendor \
    && echo "b1b32db5ec8f0aca14d0c74d712fe3e4ffbf7980aef38c806f9cb8a4a3c499f4  /opt/decomposer/vendor/redis-19.0.0.tgz" \
        | sha256sum --check --strict

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
COPY gyms/toolathlon/patches/redact-task-config.patch /tmp/redact-task-config.patch
COPY gyms/toolathlon/patches/vllm-reasoning-field.patch /tmp/vllm-reasoning-field.patch
COPY gyms/toolathlon/patches/clean-python-tool.patch /tmp/clean-python-tool.patch
COPY gyms/toolathlon/patches/idempotent-overlong-tool-output.patch /tmp/idempotent-overlong-tool-output.patch

# TaskConfig contains task-specific API credentials. The native runner used
# to pprint it into the episode log even when debug=False.
RUN patch --batch /workspace/utils/task_runner/runner.py \
        < /tmp/redact-task-config.patch

# vLLM's Gemma reasoning parser returns ``message.reasoning``.  Toolathlon's
# converter only recognizes the older reasoning_content/reasoning_details names
# and would otherwise discard every thinking trace between tool calls.
RUN patch --batch /workspace/utils/api_model/model_provider.py \
        < /tmp/vllm-reasoning-field.patch

# MCP results pass through both MCPUtil and the generic function-tool runner.
# Make overlong-output persistence idempotent so the second layer does not
# replace a full-result handle with a handle to truncated, invalid JSON.
RUN patch --batch /workspace/utils/openai_agents_monkey_patch/custom_run_impl.py \
        < /tmp/idempotent-overlong-tool-output.patch

# Local Python is an ephemeral execution tool.  Its implementation must not
# leave .python_tmp scripts in agent workspaces that exact-tree evaluators then
# mistake for model output.
RUN patch --batch /workspace/utils/aux_tools/python_interpretor.py \
        < /tmp/clean-python-tool.patch

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

# Keep volatile provenance at the end so a new commit label does not
# invalidate every expensive dependency and task-image layer.
ARG DECOMPOSER_REVISION=unknown
ARG TOOLATHLON_REVISION=unknown
LABEL org.opencontainers.image.revision="${DECOMPOSER_REVISION}" \
      io.decomposer.toolathlon.revision="${TOOLATHLON_REVISION}"
