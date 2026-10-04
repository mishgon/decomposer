ARG TOOLATHLON_BASE_IMAGE=docker.io/lockon0927/toolathlon-task-image:1016beta
FROM ${TOOLATHLON_BASE_IMAGE}

ARG PYTHON_VERSION=3.12
ENV AGENT_VENV=/opt/agents
ENV RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1

# The base image predates the pinned benchmark; keep its .venv and lockfile.
COPY external/toolathlon/scripts/ /workspace/scripts/
COPY external/toolathlon/configs/ /workspace/configs/
COPY external/toolathlon/utils/ /workspace/utils/
COPY external/toolathlon/main.py /workspace/main.py
COPY external/toolathlon/tasks/ /workspace/tasks/
COPY external/toolathlon/deployment/ /workspace/deployment/
COPY external/toolathlon/global_preparation/check_installation.py /workspace/global_preparation/check_installation.py
COPY external/toolathlon/local_binary/ /workspace/local_binary/

# python_execute must not leave files in the graded workspace.
COPY gyms/toolathlon_bench/patches/clean-python-tool.patch /tmp/
RUN patch --batch /workspace/utils/aux_tools/python_interpretor.py \
        < /tmp/clean-python-tool.patch

# The railway site changed its bootstrap page; MCP tools and schemas stay the same.
COPY gyms/toolathlon_bench/patches/12306-mcp-cookie.patch /tmp/
RUN sed -i \
        -e 's#https://kyfw.12306.cn/otn/lcQuery/init#https://kyfw.12306.cn/otn/leftTicket/init#g' \
        -e 's#html.match(/ var lc_search_url#html.match(/\\s*var lc_search_url#' \
        /workspace/node_modules/12306-mcp/build/index.js \
    && patch --batch /workspace/node_modules/12306-mcp/build/index.js \
        < /tmp/12306-mcp-cookie.patch

# Pin the Redis chart instead of resolving the Bitnami repository per task.
COPY gyms/toolathlon_bench/patches/k8s-redis-local-chart.patch /tmp/
RUN mkdir -p /opt/decomposer/vendor \
    && helm pull oci://registry-1.docker.io/bitnamicharts/redis \
        --version 19.0.0 \
        --destination /opt/decomposer/vendor \
    && echo "b1b32db5ec8f0aca14d0c74d712fe3e4ffbf7980aef38c806f9cb8a4a3c499f4  /opt/decomposer/vendor/redis-19.0.0.tgz" \
        | sha256sum --check --strict \
    && patch --batch /workspace/tasks/finalpool/k8s-redis-helm-upgrade/scripts/init_redis_helm.sh \
        < /tmp/k8s-redis-local-chart.patch

COPY gyms/toolathlon_bench/requirements.txt /tmp/agent-requirements.txt
COPY pyproject.toml README.md /tmp/decomposer/
COPY src/decomposer /tmp/decomposer/src/decomposer

RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" "${AGENT_VENV}" \
    && uv pip install \
        --python "${AGENT_VENV}/bin/python" \
        --requirements /tmp/agent-requirements.txt \
        /tmp/decomposer

COPY gyms/toolathlon_bench /opt/decomposer/gyms/toolathlon_bench

WORKDIR /workspace
# The runner drives preprocess, the Agent Server and evaluation with docker exec.
CMD ["sleep", "infinity"]
