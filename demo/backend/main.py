"""SentinelEdge Demo Backend - Real-time fraud detection server."""

import asyncio
import json
import re
import time
import os
import sys
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path so sentinel_edge package is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

app = FastAPI(title="SentinelEdge Demo", version="0.1.0")

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

        transcript_path = os.path.join(os.path.dirname(__file__), call_info["file"])
        sentences = load_transcript(transcript_path)

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
