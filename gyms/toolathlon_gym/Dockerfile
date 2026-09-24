ARG TOOLATHLON_BASE_IMAGE=toolathlon-pack:latest
FROM ${TOOLATHLON_BASE_IMAGE}

ARG PYTHON_VERSION=3.12
ENV SUBAGENT_VENV=/opt/subagents
ENV LOCAL_SERVERS_PATH=/opt/local_servers

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ffmpeg \
        fonts-liberation \
        fonts-noto-cjk \
        libreoffice-impress \
        libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

RUN set -e; for server in mcp-canvas-lms notion-mcp-server woocommerce-mcp; do \
        path="/opt/local_servers/$server"; \
        npm install --ignore-scripts --prefix "$path" pg; \
        npm install --ignore-scripts --save-dev --prefix "$path" @types/pg; \
    done; \
    npm install --omit=dev --ignore-scripts \
        --prefix /opt/local_servers/filesystem

COPY gyms/toolathlon_gym/patches/mcp-canvas-lms-postgres.patch /tmp/
RUN cd /opt/local_servers/mcp-canvas-lms \
    && git apply --no-index /tmp/mcp-canvas-lms-postgres.patch

RUN set -e; for server in \
        12306-mcp \
        Calendar-Autoauth-MCP-Server \
        HowToCook-mcp \
        google-forms-mcp \
        mcp-canvas-lms \
        mcp-npx-fetch \
        notion-mcp-server \
        playwright-mcp \
        servers \
        woocommerce-mcp \
        youtube-mcp-server; do \
        path="/opt/local_servers/$server"; \
        npm install --ignore-scripts --prefix "$path"; \
        npm run build --prefix "$path"; \
        npm prune --omit=dev --ignore-scripts --prefix "$path"; \
    done

RUN uv sync --directory /opt/local_servers/mcp-google-sheets
RUN uv pip install \
    --python /opt/local_servers/yahoo-finance-mcp/.venv/bin/python \
    psycopg2-binary
RUN uv add \
    --directory /opt/local_servers/mcp-snowflake-server \
    "mcp==1.29.0"
RUN uv pip install \
    --python /opt/venv/bin/python \
    "matplotlib==3.10.3" \
    "pandas==2.2.3" \
    "PyPDF2==3.0.1" \
    "reportlab>=4.4.3"

COPY gyms/toolathlon_gym/subagents/requirements.txt /tmp/subagent-requirements.txt
COPY pyproject.toml README.md /tmp/decomposer/
COPY src/decomposer /tmp/decomposer/src/decomposer

RUN uv python install "${PYTHON_VERSION}" \
    && uv venv --python "${PYTHON_VERSION}" "${SUBAGENT_VENV}" \
    && uv pip install \
        --python "${SUBAGENT_VENV}/bin/python" \
        --requirements /tmp/subagent-requirements.txt \
        /tmp/decomposer

COPY gyms/toolathlon_gym /opt/decomposer/gyms/toolathlon_gym

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/python3", "/opt/decomposer/gyms/toolathlon_gym/task.py"]
CMD ["serve"]
