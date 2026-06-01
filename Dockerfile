# SentinelEdge Demo - Fly.io / container deployment
# Serves React frontend + FastAPI backend from a single container

FROM node:20-slim AS frontend-build

WORKDIR /build
COPY demo/frontend/package.json demo/frontend/package-lock.json* ./
RUN npm install --production=false
COPY demo/frontend/ .
RUN npm run build

# --- Python runtime ---
FROM python:3.11-slim

# System deps for PyNaCl (libsodium)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsodium-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only the deps we actually need for the demo (no whisper/pyaudio/onnx)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy project code
COPY sentinel_edge/ sentinel_edge/
COPY hub/ hub/
COPY demo/backend/ demo/backend/
COPY models/ models/
COPY federated/ federated/

# Copy built frontend into a static dir the backend will serve
COPY --from=frontend-build /build/dist /app/static

# Copy the deployment server entrypoint
COPY deploy_server.py .

# Fly.io routes to the process port configured in fly.toml.
ENV PORT=8080
EXPOSE 8080

CMD ["python", "deploy_server.py"]
