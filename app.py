"""
FastAPI Web Demo cho TTS Model
Chạy: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import uuid
from pathlib import Path

from TTS.utils.synthesizer import Synthesizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="TTS Demo", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Thư mục audio tổng hợp ───────────────────────────────────────────────────
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(TEMP_AUDIO_DIR)), name="audio")

# ── Thư mục wavs gốc ─────────────────────────────────────────────────────────
ORIGINAL_WAVS_DIR = Path("/home/anhht/textToSpeechCuoiCham/dataset/wavs")
if ORIGINAL_WAVS_DIR.exists():
    app.mount("/original_audio", StaticFiles(directory=str(ORIGINAL_WAVS_DIR)), name="original_audio")
    print(f"[INFO] Mounted original wavs: {ORIGINAL_WAVS_DIR}")
else:
    print(f"[WARN] Thư mục wavs gốc không tồn tại: {ORIGINAL_WAVS_DIR}")

# ── Load metadata.csv ─────────────────────────────────────────────────────────
# Format mỗi dòng: crdo-TOU_VOC9_W382|hwiːt⁷|hwiːt⁷  (pipe-separated, no header)
METADATA_PATH = Path("/home/anhht/textToSpeechCuoiCham/dataset/metadata.csv")
_metadata: dict = {}   # { "phoneme_text" -> "crdo-TOU_VOC9_W382.wav" }

def load_metadata():
    global _metadata
    if not METADATA_PATH.exists():
        print(f"[WARN] metadata.csv không tồn tại: {METADATA_PATH}")
        return
    count = 0
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            file_id  = parts[0].strip()          # crdo-TOU_VOC9_W382
            phoneme  = parts[1].strip()          # hwiːt⁷
            _metadata[phoneme] = f"{file_id}.wav"
            count += 1
    print(f"[INFO] Loaded {count} entries từ metadata.csv")

load_metadata()

# ── TTS Model ─────────────────────────────────────────────────────────────────
_DEFAULT_MODEL  = "best_model/checkpoint_1220000.pth"
_DEFAULT_CONFIG = "best_model/config.json"

print("[INFO] Đang load TTS model...")
try:
    _synthesizer = Synthesizer(
        tts_checkpoint=_DEFAULT_MODEL,
        tts_config_path=_DEFAULT_CONFIG,
        use_cuda=False,
    )
    print("[INFO] Load model thành công.")
except Exception as e:
    print(f"[ERROR] Không thể load model: {e}")
    _synthesizer = None


# ── Schemas ───────────────────────────────────────────────────────────────────
class SynthesizeRequest(BaseModel):
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("index.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html không tồn tại")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    if _synthesizer is None:
        raise HTTPException(status_code=503, detail="Model chưa load được. Kiểm tra log server.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Văn bản đầu vào không được rỗng.")

    # ── Sinh audio AI ──────────────────────────────────────────────────────
    file_id         = uuid.uuid4().hex[:8]
    output_filename = f"tts_{file_id}.wav"
    output_path     = TEMP_AUDIO_DIR / output_filename

    try:
        wav = _synthesizer.tts(text)
        _synthesizer.save_wav(wav, str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tổng hợp giọng nói: {e}")

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Không sinh được file âm thanh.")

    # ── Tra cứu file gốc trong metadata ────────────────────────────────────
    original_audio_url = None
    original_filename  = None

    if text in _metadata:
        wav_name = _metadata[text]
        wav_file = ORIGINAL_WAVS_DIR / wav_name
        if wav_file.exists():
            original_audio_url = f"/original_audio/{wav_name}"
            original_filename  = wav_name

    return {
        "status":             "success",
        "audio_url":          f"/audio/{output_filename}",
        "filename":           output_filename,
        "text":               text,
        "original_audio_url": original_audio_url,   # None nếu không tìm thấy
        "original_filename":  original_filename,
        "has_original":       original_audio_url is not None,
    }


@app.delete("/audio/{filename}")
async def delete_audio(filename: str):
    file_path = TEMP_AUDIO_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="File không tồn tại.")


@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "model":            "loaded" if _synthesizer else "failed",
        "metadata_entries": len(_metadata),
    }