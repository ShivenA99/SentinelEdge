"""SentinelEdge Demo Backend - Real-time fraud detection server."""

import asyncio
import json
import re
import time
import os
import sys
import re
import urllib.request
import urllib.error
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars only

# Add project root to path so sentinel_edge package is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

app = FastAPI(title="SentinelEdge Demo", version="0.1.0")

INTERACTIVE_REPLY_TIMEOUT_SECONDS = 15.0
MIN_SCAMMER_TURNS_FOR_ALERT = 3
MIN_USER_REPLIES_FOR_ALERT = 2
MIN_CALL_SECONDS_FOR_ALERT = 20.0
REPLY_LISTEN_START_DELAY_SECONDS = 0.25
VOICE_ACTIVITY_RMS_THRESHOLD = 0.0015
MIN_VOICED_SECONDS_FOR_TRANSCRIBE = 0.3
END_OF_UTTERANCE_SILENCE_SECONDS = 0.8
MAX_UTTERANCE_SECONDS = 7.0
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


def _read_anthropic_api_key_from_env() -> str | None:
    """Read optional Anthropic key from env without hardcoding secrets."""
    key = os.getenv("SENTINEL_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if key is None:
        return None
    key = key.strip()
    return key if key else None


def _extract_anthropic_text(response_json: dict) -> str | None:
    """Extract first text block from Anthropic messages response."""
    content = response_json.get("content")
    if not isinstance(content, list):
        return None

    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _candidate_anthropic_models(configured_model: str) -> list[str]:
    """Return ordered, de-duplicated Anthropic model candidates."""
    candidates = [
        configured_model,
        "claude-sonnet-4-5-latest",
        "claude-sonnet-4-0",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
    ]

    seen: set[str] = set()
    ordered: list[str] = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _personalize_scammer_line_sync(
    *,
    base_sentence: str,
    call_description: str,
    recent_user_replies: list[str],
    recent_scammer_lines: list[str],
) -> str | None:
    """Call Anthropic API and return one personalized scammer sentence."""
    api_key = _read_anthropic_api_key_from_env()
    if api_key is None:
        print("[Claude] API key not found in env; personalization disabled.")
        return None

    # Redact key for logging (show only last 8 chars)
    key_display = f"...{api_key[-8:]}" if len(api_key) > 8 else "***"
    print(f"[Claude] API key loaded: {key_display}")
    
    configured_model = os.getenv("SENTINEL_ANTHROPIC_MODEL", "claude-sonnet-4-5-latest")
    model_candidates = _candidate_anthropic_models(configured_model)
    print(f"[Claude] Attempting personalization with model candidates: {model_candidates}")
    user_context = " | ".join(recent_user_replies[-3:])
    scammer_context = " | ".join(recent_scammer_lines[-2:])

    system_prompt = (
        "You are generating a scam call simulation sentence for cybersecurity training. "
        "Return exactly one sentence, under 35 words, plain text only. "
        "Keep it plausible for a phone scam and adapt to the victim reply context. "
        "Do not include markdown, bullet points, labels, or safety disclaimers."
    )

    user_prompt = (
        f"Call scenario: {call_description}. "
        f"Script baseline sentence: {base_sentence} "
        f"Recent scammer lines: {scammer_context or 'none'} "
        f"Recent victim replies: {user_context or 'none'}"
    )

    payload = {
        "model": model_candidates[0],
        "max_tokens": 80,
        "temperature": 0.7,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
        "content-type": "application/json",
    }

    for model in model_candidates:
        payload["model"] = model
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            print(f"[Claude] API request to {ANTHROPIC_API_URL} using model: {model}")
            with urllib.request.urlopen(request, timeout=8) as response:
                response_json = json.loads(response.read().decode("utf-8"))

            text = _extract_anthropic_text(response_json)
            if text is None:
                print(f"[Claude] Model {model} returned empty text block.")
                continue

            personalized = " ".join(text.split())
            print(f"[Claude] Personalized sentence ({model}): {personalized[:80]}...")
            return personalized
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            print(f"[Claude] HTTP Error {e.code} on model {model}: {e.reason}")
            print(f"[Claude] Response body: {error_body[:200]}")

            is_model_not_found = (
                e.code == 404
                and "not_found_error" in error_body
                and "model:" in error_body
            )
            if is_model_not_found:
                print(f"[Claude] Model not found: {model}. Trying next fallback.")
                continue
            return None
        except urllib.error.URLError as e:
            print(f"[Claude] URL Error on model {model}: {e.reason}")
            return None
        except (TimeoutError, json.JSONDecodeError) as e:
            print(f"[Claude] Request error on model {model}: {e}")
            return None

    print("[Claude] No usable Anthropic model found; personalization disabled for this turn.")
    return None


async def _personalize_scammer_line(
    *,
    base_sentence: str,
    call_description: str,
    recent_user_replies: list[str],
    recent_scammer_lines: list[str],
) -> str | None:
    """Async wrapper for sentence personalization."""
    return await asyncio.to_thread(
        _personalize_scammer_line_sync,
        base_sentence=base_sentence,
        call_description=call_description,
        recent_user_replies=recent_user_replies,
        recent_scammer_lines=recent_scammer_lines,
    )


def _read_input_device_from_env() -> str | int | None:
    """Read optional microphone input device from SENTINEL_INPUT_DEVICE.

    Supports either an integer device index or a device name string.
    """
    value = os.getenv("SENTINEL_INPUT_DEVICE")
    if value is None or not value.strip():
        return None

    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def _parse_input_device_query(value: str | None) -> str | int | None:
    """Parse optional input device from websocket query parameter."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return value

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track active WebSocket connections
active_connections: list[WebSocket] = []

# ---------------------------------------------------------------------------
# Available demo calls
# ---------------------------------------------------------------------------

SAMPLE_CALLS = {
    "live_mic": {
        "file": None,
        "caller": "Live microphone",
        "caller_name": "Local Device Input",
        "description": "Real-time microphone detection",
    },
    "irs_scam": {
        "file": "sample_calls/irs_scam.txt",
        "caller": "+1 (800) 555-0199",
        "caller_name": "Unknown Number",
        "description": "IRS Impersonation Scam",
    },
    "tech_support": {
        "file": "sample_calls/tech_support_scam.txt",
        "caller": "+1 (888) 555-0147",
        "caller_name": "Microsoft Support",
        "description": "Tech Support Scam",
    },
    "bank_fraud": {
        "file": "sample_calls/bank_fraud_scam.txt",
        "caller": "+1 (800) 555-0123",
        "caller_name": "Bank Security",
        "description": "Bank Fraud Scam",
    },
    "legitimate": {
        "file": "sample_calls/legitimate_call.txt",
        "caller": "+1 (555) 234-5678",
        "caller_name": "Dr. Smith Office",
        "description": "Legitimate Appointment Reminder",
    },
    "crypto_investment": {
        "file": "sample_calls/crypto_investment_scam.txt",
        "caller": "+1 (833) 555-0291",
        "caller_name": "Digital Asset Partners",
        "description": "Crypto Investment Scam",
    },
    "grandparent": {
        "file": "sample_calls/grandparent_scam.txt",
        "caller": "+1 (555) 867-5309",
        "caller_name": "Unknown Number",
        "description": "Grandparent Scam",
    },
    "amazon_refund": {
        "file": "sample_calls/amazon_refund_scam.txt",
        "caller": "+1 (888) 555-0342",
        "caller_name": "Amazon Support",
        "description": "Amazon Refund Scam",
    },
    "utility_shutoff": {
        "file": "sample_calls/utility_shutoff_scam.txt",
        "caller": "+1 (800) 555-0476",
        "caller_name": "Utility Company",
        "description": "Utility Shutoff Scam",
    },
    "prize_notification": {
        "file": "sample_calls/prize_notification_scam.txt",
        "caller": "+1 (877) 555-0188",
        "caller_name": "National Sweepstakes",
        "description": "Prize Notification Scam",
    },
}

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/calls")
async def list_calls():
    """List available demo calls."""
    return {"calls": [{"id": k, **v} for k, v in SAMPLE_CALLS.items()]}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "active_connections": len(active_connections)}


@app.get("/api/audio-devices")
async def list_audio_devices():
    """List available input audio devices (when sounddevice is installed)."""
    try:
        import sounddevice as sd

        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append(
                    {
                        "index": idx,
                        "name": dev.get("name", "unknown"),
                        "max_input_channels": int(dev.get("max_input_channels", 0)),
                        "default_samplerate": float(dev.get("default_samplerate", 0.0)),
                    }
                )
        return {"devices": devices}
    except Exception as exc:
        return {"devices": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# WebSocket: real-time call fraud detection
# ---------------------------------------------------------------------------


@app.websocket("/ws/call/{call_id}")
async def call_detection(websocket: WebSocket, call_id: str):
    """WebSocket endpoint for real-time call fraud detection."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        if call_id not in SAMPLE_CALLS:
            await websocket.send_json({"error": f"Unknown call: {call_id}"})
            return

        call_info = SAMPLE_CALLS[call_id]

        await websocket.send_json(
            {
                "type": "call_start",
                "caller": call_info["caller"],
                "caller_name": call_info["caller_name"],
                "description": call_info["description"],
                "timestamp": time.time(),
            }
        )

        input_device = _parse_input_device_query(
            websocket.query_params.get("input_device")
        )

        if call_id == "live_mic":
            await run_live_mic_detection(websocket, input_device=input_device)
            return

        # Load transcript sentences
        transcript_path = os.path.join(os.path.dirname(__file__), call_info["file"])
        sentences = load_transcript(transcript_path)

        interactive = websocket.query_params.get("interactive") == "1"
        if interactive:
            await run_interactive_scripted_call(
                websocket,
                sentences,
                input_device=input_device,
            )
            return

        # Import detection pipeline components
        from sentinel_edge.features.handcrafted import extract_handcrafted_features
        from sentinel_edge.features.feature_pipeline import FeaturePipeline
        from sentinel_edge.classifier.xgb_classifier import FraudClassifier
        from sentinel_edge.classifier.score_accumulator import ScoreAccumulator
        from sentinel_edge.classifier.alert_engine import AlertEngine

        accumulator = ScoreAccumulator(alpha=0.3)
        alert_engine = AlertEngine()

        _model_path = os.path.join(_PROJECT_ROOT, "models", "call_fraud_xgb.json")
        _tfidf_path = os.path.join(_PROJECT_ROOT, "models", "tfidf_call_vectorizer.pkl")
        _use_real_model = os.path.exists(_model_path) and os.path.exists(_tfidf_path)
        if _use_real_model:
            _classifier = FraudClassifier(_model_path)
            _pipeline = FeaturePipeline(_tfidf_path)
        else:
            _classifier = None
            _pipeline = None

        call_start_time = time.time()

        for i, sentence in enumerate(sentences):
            await asyncio.sleep(1.5 + np.random.random() * 1.5)

            features = extract_handcrafted_features(sentence)

            if _use_real_model:
                t0 = time.perf_counter()
                feature_vec = _pipeline.extract(sentence)
                fraud_score = _classifier.predict_proba(feature_vec)
                _inference_ms = (time.perf_counter() - t0) * 1000
            else:
                fraud_score = compute_heuristic_score(features)
                _inference_ms = np.random.uniform(5, 15)

            ema_score = accumulator.update(fraud_score)
            alert = alert_engine.evaluate(ema_score, features)
            elapsed = time.time() - call_start_time

            await websocket.send_json(
                {
                    "type": "sentence",
                    "speaker": "scammer",
                    "index": i,
                    "text": sentence,
                    "raw_score": round(fraud_score, 4),
                    "ema_score": round(ema_score, 4),
                    "features": {
                        k: round(v, 4) if isinstance(v, float) else int(v)
                        for k, v in features.items()
                    },
                    "alert": {
                        "should_alert": alert.should_alert,
                        "risk_level": alert.risk_level.value,
                        "reasons": alert.reasons,
                    },
                    "elapsed_seconds": round(elapsed, 1),
                    "inference_ms": round(_inference_ms, 1),
                    "timestamp": time.time(),
                }
            )

            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.01
                )
                data = json.loads(msg)
                if data.get("action") == "block":
                    await websocket.send_json(
                        {"type": "call_blocked", "timestamp": time.time()}
                    )
                    return
                elif data.get("action") == "dismiss":
                    pass
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

        await websocket.send_json(
            {
                "type": "call_end",
                "final_score": round(accumulator.current_score, 4),
                "peak_score": round(accumulator.peak_score, 4),
                "mean_score": round(accumulator.mean_score, 4),
                "total_sentences": len(sentences),
                "duration_seconds": round(time.time() - call_start_time, 1),
                "timestamp": time.time(),
            }
        )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json(
                {"type": "error", "message": str(exc), "timestamp": time.time()}
            )
        except Exception:
            pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


async def run_interactive_scripted_call(
    websocket: WebSocket,
    scripted_sentences: list[str],
    input_device: str | int | None = None,
) -> None:
    """Turn-based scripted call: scammer line, wait for user reply, repeat."""
    try:
        from live_mic import LiveMicCapture
    except ImportError:
        from demo.backend.live_mic import LiveMicCapture
    from sentinel_edge.audio.transcriber import Transcriber
    from sentinel_edge.engine import SentinelEngine

    engine = SentinelEngine(models_dir=os.path.join(_PROJECT_ROOT, "models"))
    engine.reset_call_state()
    mic = LiveMicCapture(
        sample_rate=16_000,
        chunk_size=2048,
        channels=1,
        input_device=(input_device if input_device is not None else _read_input_device_from_env()),
    )
    whisper_model = os.getenv("SENTINEL_WHISPER_MODEL", "base.en")
    transcriber = Transcriber(model_name=whisper_model)

    call_start_time = time.time()
    turn_index = 0
    scammer_turn_count = 0
    user_reply_count = 0
    fraud_alert_sent = False
    user_replies: list[str] = []
    scammer_history: list[str] = []
    last_scammer_sentence = ""

    try:
        mic.start()

        for scam_sentence in scripted_sentences:
            await asyncio.sleep(1.0)

            line_to_send = scam_sentence
            if user_replies:
                print(f"[Main] Attempting personalization (user has {len(user_replies)} replies)")
                personalized = await _personalize_scammer_line(
                    base_sentence=scam_sentence,
                    call_description="Interactive scam call simulation",
                    recent_user_replies=user_replies,
                    recent_scammer_lines=scammer_history,
                )
                if personalized:
                    line_to_send = personalized
            else:
                print(f"[Main] No user replies yet; skipping personalization")

            t0 = time.perf_counter()
            fraud_score, features = engine.analyze_sentence(line_to_send)
            ema_score = engine.accumulator.update(fraud_score)
            alert = engine.alert_engine.evaluate(ema_score, features)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            await websocket.send_json(
                {
                    "type": "sentence",
                    "speaker": "scammer",
                    "index": turn_index,
                    "text": line_to_send,
                    "raw_score": round(fraud_score, 4),
                    "ema_score": round(ema_score, 4),
                    "features": {
                        k: round(v, 4) if isinstance(v, float) else int(v)
                        for k, v in features.items()
                    },
                    "alert": {
                        "should_alert": alert.should_alert,
                        "risk_level": alert.risk_level.value,
                        "reasons": alert.reasons,
                    },
                    "elapsed_seconds": round(time.time() - call_start_time, 1),
                    "inference_ms": round(inference_ms, 2),
                    "timestamp": time.time(),
                }
            )
            turn_index += 1
            scammer_turn_count += 1
            last_scammer_sentence = line_to_send
            scammer_history.append(line_to_send)

            elapsed_seconds = time.time() - call_start_time
            if (
                not fraud_alert_sent
                and alert.should_alert
                and scammer_turn_count >= MIN_SCAMMER_TURNS_FOR_ALERT
                and user_reply_count >= MIN_USER_REPLIES_FOR_ALERT
                and elapsed_seconds >= MIN_CALL_SECONDS_FOR_ALERT
            ):
                fraud_alert_sent = True
                await websocket.send_json(
                    {
                        "type": "fraud_detected",
                        "message": "High scam risk detected. Hang up now.",
                        "risk_level": alert.risk_level.value,
                        "ema_score": round(ema_score, 4),
                        "reasons": alert.reasons,
                        "timestamp": time.time(),
                    }
                )

            await websocket.send_json(
                {
                    "type": "waiting_for_reply",
                    "timeout_seconds": int(INTERACTIVE_REPLY_TIMEOUT_SECONDS),
                    "timestamp": time.time(),
                }
            )

            # Ensure only fresh user turn audio is considered.
            mic.drain_queue()

            user_sentence, heard_audio = await _capture_user_sentence(
                websocket=websocket,
                mic=mic,
                transcriber=transcriber,
                timeout_seconds=INTERACTIVE_REPLY_TIMEOUT_SECONDS,
                listen_start_delay_seconds=REPLY_LISTEN_START_DELAY_SECONDS,
                voice_activity_rms_threshold=VOICE_ACTIVITY_RMS_THRESHOLD,
                min_voiced_seconds_for_transcribe=MIN_VOICED_SECONDS_FOR_TRANSCRIBE,
            )

            if not heard_audio:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "No microphone audio detected during reply window. "
                            "Check OS microphone permission and input device selection."
                        ),
                        "timestamp": time.time(),
                    }
                )
                continue

            if user_sentence is None:
                await websocket.send_json(
                    {
                        "type": "user_timeout",
                        "message": "No reply detected, continuing call.",
                        "timestamp": time.time(),
                    }
                )
                continue

            if _looks_like_echo(user_sentence, last_scammer_sentence):
                await websocket.send_json(
                    {
                        "type": "user_echo_detected",
                        "message": "Detected speaker echo. Use headphones or lower volume and repeat.",
                        "timestamp": time.time(),
                    }
                )
                continue

            # Score user reply for display only. Do not mix into scammer EMA.
            user_score, user_features = engine.analyze_sentence(user_sentence)
            user_replies.append(user_sentence)
            user_reply_count += 1
            await websocket.send_json(
                {
                    "type": "sentence",
                    "speaker": "you",
                    "index": turn_index,
                    "text": user_sentence,
                    "raw_score": round(user_score, 4),
                    "ema_score": round(engine.accumulator.current_score, 4),
                    "features": {
                        k: round(v, 4) if isinstance(v, float) else int(v)
                        for k, v in user_features.items()
                    },
                    "alert": {
                        "should_alert": False,
                        "risk_level": "safe",
                        "reasons": [],
                    },
                    "elapsed_seconds": round(time.time() - call_start_time, 1),
                    "inference_ms": 0.0,
                    "timestamp": time.time(),
                }
            )
            turn_index += 1

            # Allow client action such as manual block/hangup.
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if data.get("action") == "block":
                    await websocket.send_json(
                        {"type": "call_blocked", "timestamp": time.time()}
                    )
                    break
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

        await websocket.send_json(
            {
                "type": "call_end",
                "final_score": round(engine.accumulator.current_score, 4),
                "peak_score": round(engine.accumulator.peak_score, 4),
                "mean_score": round(engine.accumulator.mean_score, 4),
                "total_sentences": turn_index,
                "duration_seconds": round(time.time() - call_start_time, 1),
                "timestamp": time.time(),
            }
        )
    finally:
        mic.stop()


async def _capture_user_sentence(
    websocket: WebSocket,
    mic,
    transcriber,
    timeout_seconds: float,
    listen_start_delay_seconds: float,
    voice_activity_rms_threshold: float,
    min_voiced_seconds_for_transcribe: float,
) -> tuple[str | None, bool]:
    """Capture microphone input and return one completed user sentence."""
    from sentinel_edge.audio.sentence_splitter import SentenceSplitter

    splitter = SentenceSplitter()
    utterance_chunks: list[np.ndarray] = []
    utterance_samples = 0
    min_samples = int(16_000 * min_voiced_seconds_for_transcribe)
    max_samples = int(16_000 * MAX_UTTERANCE_SECONDS)
    deadline = time.time() + timeout_seconds
    listen_start = time.time() + listen_start_delay_seconds
    heard_audio = False
    speech_started = False
    silence_run_seconds = 0.0

    while time.time() < deadline:
        # Handle immediate client actions while waiting for reply.
        try:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
            data = json.loads(msg)
            if data.get("action") == "block":
                return None, heard_audio
        except asyncio.TimeoutError:
            pass
        except json.JSONDecodeError:
            pass

        chunk = mic.get_chunk(timeout=0.2)
        if chunk is None:
            await asyncio.sleep(0.01)
            continue

        if time.time() < listen_start:
            # Avoid immediately picking up system TTS bleed-through.
            continue

        heard_audio = True

        rms = float(np.sqrt(np.mean(np.square(chunk))))
        chunk_seconds = len(chunk) / 16_000.0

        if rms < voice_activity_rms_threshold:
            if speech_started:
                # Keep trailing silence so Whisper captures final words naturally.
                utterance_chunks.append(chunk)
                utterance_samples += len(chunk)
                silence_run_seconds += chunk_seconds
                if (
                    utterance_samples >= min_samples
                    and silence_run_seconds >= END_OF_UTTERANCE_SILENCE_SECONDS
                ):
                    break
            continue

        speech_started = True
        silence_run_seconds = 0.0
        utterance_chunks.append(chunk)
        utterance_samples += len(chunk)
        if utterance_samples >= max_samples:
            break

    if utterance_chunks and utterance_samples >= min_samples:
        audio = np.concatenate(utterance_chunks)
        transcript = transcriber.transcribe(audio, sample_rate=16_000).strip()
        if transcript:
            sentences = splitter.feed(transcript)
            if sentences:
                return sentences[0], heard_audio

            # Whisper can return partial text without sentence punctuation.
            if len(transcript.split()) >= 1:
                return transcript, heard_audio

    # Fallback: if we heard audio but speech never crossed VAD, try a short best-effort
    # transcription on what we did capture to handle very quiet microphones.
    if utterance_chunks and heard_audio:
        audio = np.concatenate(utterance_chunks)
        transcript = transcriber.transcribe(audio, sample_rate=16_000).strip()
        if transcript:
            return transcript, heard_audio

    leftover = splitter.flush()
    if leftover:
        return leftover, heard_audio
    return None, heard_audio


def _looks_like_echo(user_sentence: str, scam_sentence: str) -> bool:
    """Heuristic guard: detect if user transcript likely matches scammer TTS echo."""
    def _normalize(text: str) -> list[str]:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 2]
        return tokens

    user_tokens = _normalize(user_sentence)
    scam_tokens = _normalize(scam_sentence)

    if not user_tokens or not scam_tokens:
        return False

    user_set = set(user_tokens)
    scam_set = set(scam_tokens)
    overlap = len(user_set & scam_set)
    ratio = overlap / max(len(user_set), 1)

    # Require substantial overlap to reduce false positives on short user replies.
    return overlap >= 4 and ratio >= 0.85


async def run_live_mic_detection(
    websocket: WebSocket,
    input_device: str | int | None = None,
) -> None:
    """Run live microphone -> transcription -> model scoring pipeline."""
    try:
        from live_mic import LiveMicCapture
    except ImportError:
        from demo.backend.live_mic import LiveMicCapture
    from sentinel_edge.audio.sentence_splitter import SentenceSplitter
    from sentinel_edge.audio.transcriber import Transcriber
    from sentinel_edge.engine import SentinelEngine

    mic = LiveMicCapture(
        sample_rate=16_000,
        chunk_size=2048,
        channels=1,
        input_device=(input_device if input_device is not None else _read_input_device_from_env()),
    )
    splitter = SentenceSplitter()
    whisper_model = os.getenv("SENTINEL_WHISPER_MODEL", "base.en")
    transcriber = Transcriber(model_name=whisper_model)
    engine = SentinelEngine(models_dir=os.path.join(_PROJECT_ROOT, "models"))
    engine.reset_call_state()

    call_start_time = time.time()
    sentence_index = 0
    chunk_accumulator: list[np.ndarray] = []
    accumulated_samples = 0
    # Transcribe roughly every 2 seconds for stable Whisper context.
    target_samples = 16_000 * 2

    try:
        mic.start()

        while True:
            # Non-blocking action handling from client.
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                data = json.loads(msg)
                if data.get("action") == "block":
                    await websocket.send_json(
                        {"type": "call_blocked", "timestamp": time.time()}
                    )
                    break
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            chunk = mic.get_chunk(timeout=0.2)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            chunk_accumulator.append(chunk)
            accumulated_samples += len(chunk)

            if accumulated_samples < target_samples:
                continue

            audio = np.concatenate(chunk_accumulator)
            chunk_accumulator = []
            accumulated_samples = 0

            transcript = transcriber.transcribe(audio, sample_rate=16_000).strip()
            if not transcript:
                continue

            for sentence in splitter.feed(transcript):
                t0 = time.perf_counter()
                fraud_score, features = engine.analyze_sentence(sentence)
                ema_score = engine.accumulator.update(fraud_score)
                alert = engine.alert_engine.evaluate(ema_score, features)
                inference_ms = (time.perf_counter() - t0) * 1000.0

                await websocket.send_json(
                    {
                        "type": "sentence",
                        "index": sentence_index,
                        "text": sentence,
                        "raw_score": round(fraud_score, 4),
                        "ema_score": round(ema_score, 4),
                        "features": {
                            k: round(v, 4) if isinstance(v, float) else int(v)
                            for k, v in features.items()
                        },
                        "alert": {
                            "should_alert": alert.should_alert,
                            "risk_level": alert.risk_level.value,
                            "reasons": alert.reasons,
                        },
                        "elapsed_seconds": round(time.time() - call_start_time, 1),
                        "inference_ms": round(inference_ms, 2),
                        "timestamp": time.time(),
                    }
                )
                sentence_index += 1

        leftover = splitter.flush()
        if leftover:
            t0 = time.perf_counter()
            fraud_score, features = engine.analyze_sentence(leftover)
            ema_score = engine.accumulator.update(fraud_score)
            alert = engine.alert_engine.evaluate(ema_score, features)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            await websocket.send_json(
                {
                    "type": "sentence",
                    "index": sentence_index,
                    "text": leftover,
                    "raw_score": round(fraud_score, 4),
                    "ema_score": round(ema_score, 4),
                    "features": {
                        k: round(v, 4) if isinstance(v, float) else int(v)
                        for k, v in features.items()
                    },
                    "alert": {
                        "should_alert": alert.should_alert,
                        "risk_level": alert.risk_level.value,
                        "reasons": alert.reasons,
                    },
                    "elapsed_seconds": round(time.time() - call_start_time, 1),
                    "inference_ms": round(inference_ms, 2),
                    "timestamp": time.time(),
                }
            )
            sentence_index += 1

        await websocket.send_json(
            {
                "type": "call_end",
                "final_score": round(engine.accumulator.current_score, 4),
                "peak_score": round(engine.accumulator.peak_score, 4),
                "mean_score": round(engine.accumulator.mean_score, 4),
                "total_sentences": sentence_index,
                "duration_seconds": round(time.time() - call_start_time, 1),
                "timestamp": time.time(),
            }
        )
    finally:
        mic.stop()


# ---------------------------------------------------------------------------
# WebSocket: live microphone fraud detection
# ---------------------------------------------------------------------------

_LIVE_MIC_BUFFER_DURATION = 5.0


@app.websocket("/ws/call/live")
async def live_mic_detection(websocket: WebSocket):
    """WebSocket endpoint for live microphone fraud detection."""
    await websocket.accept()
    active_connections.append(websocket)

    mic = None
    try:
        try:
            try:
                from live_mic import LiveMicCapture
            except ImportError:
                from demo.backend.live_mic import LiveMicCapture
        except ImportError as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Missing dependency for live mic: {e}",
                "timestamp": time.time(),
            })
            return

        try:
            from sentinel_edge.audio.transcriber import Transcriber
        except ImportError as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Missing dependency for transcription: {e}",
                "timestamp": time.time(),
            })
            return

        mic = LiveMicCapture(sample_rate=16000)
        transcriber = Transcriber(model_name="tiny.en")

        if not transcriber.is_loaded:
            await websocket.send_json({
                "type": "status",
                "message": "Loading Whisper model (may download on first use)...",
                "timestamp": time.time(),
            })

        try:
            transcriber.load()
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": (
                    f"Failed to load Whisper model: {e}. "
                    "Install with: pip install openai-whisper"
                ),
                "timestamp": time.time(),
            })
            return

        from sentinel_edge.features.handcrafted import extract_handcrafted_features
        from sentinel_edge.features.feature_pipeline import FeaturePipeline
        from sentinel_edge.classifier.xgb_classifier import FraudClassifier as XGBFraudClassifier
        from sentinel_edge.classifier.score_accumulator import ScoreAccumulator
        from sentinel_edge.classifier.alert_engine import AlertEngine

        accumulator = ScoreAccumulator(alpha=0.3)
        alert_engine = AlertEngine()

        _model_path = os.path.join(_PROJECT_ROOT, "models", "call_fraud_xgb.json")
        _tfidf_path = os.path.join(_PROJECT_ROOT, "models", "tfidf_call_vectorizer.pkl")
        _use_real_model = os.path.exists(_model_path) and os.path.exists(_tfidf_path)
        if _use_real_model:
            _classifier = XGBFraudClassifier(_model_path)
            _pipeline = FeaturePipeline(_tfidf_path)
        else:
            _classifier = None
            _pipeline = None

        try:
            mic.start()
        except ImportError as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": time.time(),
            })
            return
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to start microphone: {e}",
                "timestamp": time.time(),
            })
            return

        await websocket.send_json({
            "type": "call_start",
            "caller": "Live Microphone",
            "caller_name": "Live Input",
            "description": "Live Microphone Analysis",
            "timestamp": time.time(),
        })

        call_start_time = time.time()
        sentence_index = 0
        audio_buffer: list[np.ndarray] = []
        buffered_seconds = 0.0

        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(), timeout=0.05
                )
                data = json.loads(msg)
                if data.get("action") in ("stop", "block"):
                    if data.get("action") == "block":
                        await websocket.send_json({
                            "type": "call_blocked",
                            "timestamp": time.time(),
                        })
                    break
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            chunk = await asyncio.get_event_loop().run_in_executor(
                None, mic.get_chunk, 0.1
            )
            if chunk is not None:
                audio_buffer.append(chunk)
                buffered_seconds += len(chunk) / mic.sample_rate

            if buffered_seconds >= _LIVE_MIC_BUFFER_DURATION and audio_buffer:
                audio_segment = np.concatenate(audio_buffer)
                audio_buffer.clear()
                buffered_seconds = 0.0

                transcript = await asyncio.get_event_loop().run_in_executor(
                    None, transcriber.transcribe, audio_segment, 16000
                )

                if not transcript.strip():
                    continue

                sentences = _split_sentences(transcript)

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    features = extract_handcrafted_features(sentence)

                    if _use_real_model:
                        t0 = time.perf_counter()
                        feature_vec = _pipeline.extract(sentence)
                        fraud_score = _classifier.predict_proba(feature_vec)
                        _inference_ms = (time.perf_counter() - t0) * 1000
                    else:
                        fraud_score = compute_heuristic_score(features)
                        _inference_ms = np.random.uniform(5, 15)

                    ema_score = accumulator.update(fraud_score)
                    alert = alert_engine.evaluate(ema_score, features)
                    elapsed = time.time() - call_start_time

                    await websocket.send_json({
                        "type": "sentence",
                        "index": sentence_index,
                        "text": sentence,
                        "raw_score": round(fraud_score, 4),
                        "ema_score": round(ema_score, 4),
                        "features": {
                            k: round(v, 4) if isinstance(v, float) else int(v)
                            for k, v in features.items()
                        },
                        "alert": {
                            "should_alert": alert.should_alert,
                            "risk_level": alert.risk_level.value,
                            "reasons": alert.reasons,
                        },
                        "elapsed_seconds": round(elapsed, 1),
                        "inference_ms": round(_inference_ms, 1),
                        "timestamp": time.time(),
                    })
                    sentence_index += 1

        await websocket.send_json({
            "type": "call_end",
            "final_score": round(accumulator.current_score, 4),
            "peak_score": round(accumulator.peak_score, 4),
            "mean_score": round(accumulator.mean_score, 4),
            "total_sentences": sentence_index,
            "duration_seconds": round(time.time() - call_start_time, 1),
            "timestamp": time.time(),
        })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
                "timestamp": time.time(),
            })
        except Exception:
            pass
    finally:
        if mic is not None and mic.is_running:
            mic.stop()
        if websocket in active_connections:
            active_connections.remove(websocket)


def _split_sentences(text: str) -> list[str]:
    """Split transcribed text into individual sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# WebSocket: privacy / federated learning demo
# ---------------------------------------------------------------------------


@app.websocket("/ws/privacy-demo")
async def privacy_demo(websocket: WebSocket):
    """Show what the hub sees vs what stays on device."""
    await websocket.accept()

    try:
        examples = [
            "This is the IRS calling about your unpaid tax debt of five thousand dollars.",
            "Your social security number has been suspended due to suspicious activity.",
            "Press 1 now to speak with an agent or a warrant will be issued for your arrest.",
            "You need to purchase gift cards and read us the numbers to resolve this matter.",
        ]

        from sentinel_edge.features.handcrafted import extract_handcrafted_features
        from sentinel_edge.privacy.dp_noise import DPNoiseInjector

        n_local_samples = 50
        dp = DPNoiseInjector(epsilon=0.3)

        for idx, sentence in enumerate(examples):
            await asyncio.sleep(2.0)

            features = extract_handcrafted_features(sentence)
            feature_vector = np.array(list(features.values()), dtype=np.float64)

            gradient = np.random.randn(len(feature_vector)) * 0.01
            clipped = dp.clip_gradient(gradient)
            noised_gradient = dp.add_noise(clipped, n_local_samples=n_local_samples)
            sensitivity = 1.0 / n_local_samples
            sigma = dp.compute_sigma(sensitivity)

            await websocket.send_json(
                {
                    "type": "privacy_comparison",
                    "index": idx,
                    "on_device": {
                        "transcript": sentence,
                        "features": {
                            k: round(v, 4) if isinstance(v, float) else int(v)
                            for k, v in features.items()
                        },
                        "fraud_score": round(
                            float(np.random.uniform(0.6, 0.95)), 4
                        ),
                    },
                    "hub_sees": {
                        "gradient_vector": [
                            round(float(x), 6) for x in noised_gradient[:20]
                        ],
                        "gradient_size": len(noised_gradient),
                        "dp_sigma": round(float(sigma), 6),
                        "dp_epsilon": 0.3,
                        "n_samples": n_local_samples,
                        "model_version": 3,
                    },
                    "timestamp": time.time(),
                }
            )

        await websocket.send_json({"type": "demo_complete"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json(
                {"type": "error", "message": str(exc), "timestamp": time.time()}
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Transcript loading helpers
# ---------------------------------------------------------------------------


def load_transcript(file_path: str) -> list[str]:
    """Load transcript sentences from a text file (one sentence per line)."""
    if not os.path.exists(file_path):
        return get_fallback_transcript(file_path)

    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def get_fallback_transcript(file_path: str) -> list[str]:
    """Return a built-in transcript when the text file hasn't been created yet."""
    name = os.path.basename(file_path)

    if "irs" in name:
        return [
            "Hello, this is Agent Williams from the Internal Revenue Service.",
            "We are calling because there is a problem with your tax return.",
            "Our records indicate that you owe the IRS three thousand five hundred dollars in back taxes.",
            "If this amount is not paid immediately, a warrant will be issued for your arrest.",
            "This is your final notice before legal action is taken against you.",
            "You can resolve this matter today by making a payment over the phone.",
            "We accept payment through gift cards, wire transfer, or cryptocurrency.",
            "Please provide your social security number so we can verify your identity.",
            "Time is running out. You must act now to avoid criminal prosecution.",
            "Press 1 to be connected to a payment specialist immediately.",
        ]
    elif "tech" in name:
        return [
            "Hello, this is the Microsoft Windows technical support department.",
            "We have detected that your computer has been compromised by malicious software.",
            "Hackers are currently accessing your personal files and banking information.",
            "You need to act immediately to prevent further damage to your system.",
            "I will need you to download our remote access tool so we can fix this problem.",
            "Please go to this website and enter the access code I give you.",
            "We need to verify your identity. Can you provide your email password?",
            "There is a one-time security fee of two hundred and ninety nine dollars.",
            "You can pay with a credit card or by purchasing Google Play gift cards.",
            "If you do not resolve this now, your computer will be permanently locked.",
        ]
    elif "bank" in name:
        return [
            "Good afternoon, this is the fraud prevention department at your bank.",
            "We have detected suspicious activity on your account ending in four seven eight two.",
            "Someone attempted to make a large purchase of fifteen hundred dollars.",
            "For your security, we need to verify your identity immediately.",
            "Can you please confirm your full account number and routing number?",
            "We also need the three-digit security code on the back of your card.",
            "I'm going to send you a verification code. Please read it back to me.",
            "We need to transfer your funds to a secure account to protect them.",
            "Please do not hang up or contact your branch directly as this is time-sensitive.",
            "Your account will be frozen if we cannot complete this verification process.",
        ]
    else:
        return [
            "Hi, this is Sarah from Doctor Smith's office calling.",
            "I'm calling to confirm your appointment scheduled for next Tuesday at two thirty.",
            "Doctor Smith will be performing your annual checkup.",
            "Please remember to bring your insurance card and photo ID.",
            "Also, please arrive fifteen minutes early to fill out any updated paperwork.",
            "If you need to reschedule, please call us back at your convenience.",
            "We look forward to seeing you. Have a great day!",
        ]


# ---------------------------------------------------------------------------
# Heuristic fraud scorer (substitutes for XGBoost in demo mode)
# ---------------------------------------------------------------------------


def compute_heuristic_score(features: dict[str, float]) -> float:
    """Compute a heuristic fraud score from handcrafted features."""
    score = 0.0
    weights = {
        "urgency_count": 0.08,
        "action_count": 0.06,
        "financial_count": 0.07,
        "impersonation_count": 0.10,
        "has_url": 0.04,
        "has_shortened_url": 0.06,
        "has_verify_pattern": 0.04,
        "has_threat": 0.10,
        "has_prize": 0.06,
        "has_account_ref": 0.07,
        "dollar_sign": 0.03,
        "has_phone_number": 0.02,
        "exclamation_count": 0.02,
        "caps_ratio": 0.03,
    }

    for feature, weight in weights.items():
        value = features.get(feature, 0)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        score += value * weight

    score = max(0.0, min(1.0, score))
    score += np.random.normal(0, 0.02)
    score = max(0.0, min(1.0, score))

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[os.path.dirname(__file__)],
    )
