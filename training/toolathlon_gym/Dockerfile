ARG GYM_IMAGE=localhost/decomposer-toolathlon:latest
FROM ${GYM_IMAGE}
COPY gyms/toolathlon_gym/patches/postgres-services.patch /tmp/postgres-services.patch
RUN cd /opt && git apply --no-index --check /tmp/postgres-services.patch \
    && git apply --no-index /tmp/postgres-services.patch \
    && npm ci --include=dev --ignore-scripts --prefix /opt/local_servers/mcp-canvas-lms \
    && npm ci --include=dev --ignore-scripts --prefix /opt/local_servers/woocommerce-mcp \
    && npm ci --include=dev --ignore-scripts --prefix /opt/local_servers/notion-mcp-server \
    && npm run build --prefix /opt/local_servers/woocommerce-mcp \
    && npm run build --prefix /opt/local_servers/notion-mcp-server \
    && npm run build --prefix /opt/local_servers/mcp-canvas-lms \
    && npm prune --omit=dev --ignore-scripts --prefix /opt/local_servers/woocommerce-mcp \
    && npm prune --omit=dev --ignore-scripts --prefix /opt/local_servers/notion-mcp-server \
    && npm prune --omit=dev --ignore-scripts --prefix /opt/local_servers/mcp-canvas-lms
