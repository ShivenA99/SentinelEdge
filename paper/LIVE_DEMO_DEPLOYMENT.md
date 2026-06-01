# Live Demo Deployment Guide

EMNLP 2026 demos require a public live demo URL or installable
package link in the paper, and this requirement is **strictly
enforced** (submissions without one are desk-rejected). The link
must remain reachable from at least submission day (July 10, 2026)
through camera-ready (August 30) and ideally through the conference
(October 24-29).

This guide describes how to host SentinelEdge's web demo cheaply
and reliably for that window.

## Option A (recommended): Fly.io / Railway / Render

Single-container deployment of the FastAPI backend plus the built
React frontend. Cost: roughly $5-10/month for the minimum tier.

### One-time setup

1. **Build the frontend** to static files:

   ```bash
   cd demo/frontend
   npm install
   npm run build
   # Produces demo/frontend/dist/
   ```

2. **Update the backend** to serve the static files. Add to
   `demo/backend/main.py`:

   ```python
   from fastapi.staticfiles import StaticFiles
   app.mount("/", StaticFiles(
       directory="../frontend/dist", html=True
   ), name="frontend")
   ```

   (Mount last, after API routes, so `/ws` etc. still take
   precedence.)

3. **Add a `Dockerfile`** at the repo root:

   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   # Build the frontend in a multi-stage build, or pre-build and copy dist/
   EXPOSE 8000
   CMD ["uvicorn", "demo.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

4. **Deploy** with Fly.io (or your platform of choice):

   ```bash
   curl -L https://fly.io/install.sh | sh
   fly launch          # answers: no Postgres, no Redis, run only once
   fly deploy
   ```

   Fly will return a URL like `https://sentineledge-demo.fly.dev`.

5. **Test from incognito**: open the URL on a fresh browser session,
   confirm the demo loads, click through all nine pre-loaded calls.

### Persistent uptime considerations

- Fly's free tier auto-scales to zero when idle. Set
  `min_machines_running = 1` in `fly.toml` so reviewers don't hit a
  30-s cold start. This costs a few dollars more per month but is
  worth it for the review window.
- Set the platform's health check to hit `/healthz` (or any GET
  endpoint that responds 200). Add such an endpoint to the backend
  if not present:

  ```python
  @app.get("/healthz")
  def healthz():
      return {"ok": True, "model_loaded": True}
  ```

- Configure log retention. Some platforms drop logs after 24 h on
  free tiers; this is fine but be aware if you want to debug
  reviewer behaviour.

## Option B (cheapest): a small VPS with systemd

DigitalOcean, Hetzner, or Linode at $4-6/month.

```bash
# On the VPS, as a non-root user
git clone <your-repo>
cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd demo/frontend && npm install && npm run build && cd ../..

# Systemd unit
sudo tee /etc/systemd/system/sentineledge.service <<EOF
[Unit]
Description=SentinelEdge Demo
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.venv/bin
ExecStart=$(pwd)/.venv/bin/uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now sentineledge

# Caddy in front, for HTTPS:
sudo apt install caddy
sudo tee /etc/caddy/Caddyfile <<EOF
demo.sentineledge.example.com {
    reverse_proxy localhost:8000
}
EOF
sudo systemctl restart caddy
```

Caddy gets you Let's Encrypt HTTPS automatically. You need a
domain; cheap options are Cloudflare Registrar or Porkbun ($10/yr).

## Option C (fallback): installable package, no live URL

If hosting genuinely won't work (budget, institutional firewall,
data-privacy review pending), the demo call permits an "installable
package" link instead. State this explicitly in the paper and
provide:

- A tagged GitHub release with a pre-built artefact
- A one-command setup that brings the demo up locally:

  ```bash
  pip install sentineledge-demo
  sentineledge-demo            # opens http://localhost:8000
  ```

- A `README.md` at the release page that walks a reviewer through
  the setup in under five minutes

This is acceptable but inferior; reviewers strongly prefer a live
URL.

## What the demo must guarantee

For each of the nine pre-loaded calls, the live demo must:

1. Run the actual trained model (the heuristic fallback is now
   removed -- see Section 3 of the paper).
2. Show a streaming EMA trajectory.
3. Fire an alert exactly when the EMA crosses 0.75.
4. Display real `time.perf_counter()` per-sentence latency, not
   simulated.
5. Function without external network calls (no OpenAI / Anthropic
   API in the live path).
6. Work in an incognito browser session.
7. Not crash if the visitor pastes their own transcript, including
   edge cases like an empty string or a 5000-character paragraph.

## Rate-limit and abuse protection

A public demo will eventually attract scraping or denial-of-service.
Minimum protections:

- A per-IP rate limit on the WebSocket endpoint (e.g.\ 10
  concurrent calls per IP).
- A maximum-transcript-length check (reject > 10 KB strings).
- A short-circuit for inputs that don't contain text (don't even
  feature-extract empty / whitespace-only input).
- Log every request to a rolling file; if abuse appears, switch to
  Cloudflare in front of the origin.

The backend has none of this by default. Add at least the per-IP
limit before publishing the URL. A minimal implementation with
`slowapi`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.websocket("/ws")
@limiter.limit("10/minute")
async def ws_endpoint(websocket: WebSocket):
    ...
```

## Cost estimate, end-to-end

| Component | Cost / month | Notes |
|---|---|---|
| Fly.io minimum tier with 1 always-on machine | $5-10 | scales with traffic |
| Domain (one-time, optional) | $1/month equivalent | $10-15/year on Cloudflare/Porkbun |
| Cloudflare DNS + WAF | free tier | optional but recommended |
| **Total** | **~$10/month** | through October 2026 = ~$50 total |

## Submission day

1. The URL must be in the paper PDF (already a placeholder in
   `\maketitle`).
2. The URL must also go in the OpenReview submission form (there is
   a separate field).
3. Confirm reachability with `curl -I https://your-url/healthz` on
   submission day.
4. After acceptance: keep it running through camera-ready and
   conference. After the conference, you can take it down or
   continue hosting as you choose.

## What to do if the demo goes down during review

EMNLP demo reviews happen between July 10 and August 20. If the
URL is unreachable when a reviewer clicks it, the paper is at risk.
Mitigations:

- Set up uptime monitoring (UptimeRobot is free) on the URL so you
  get email/SMS when it drops.
- Have a tagged Docker image as a backup deployment artefact.
- If you must take the URL offline temporarily, post a one-line
  notice at the URL ("demo briefly offline, back by HH:MM UTC")
  rather than letting it 502.
