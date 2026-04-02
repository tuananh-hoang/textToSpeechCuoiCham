### demo: https://huggingface.co/spaces/stephenhoang/tts-cuoi-cham-demo

---

## 1. THÔNG TIN TỔNG QUAN

| Trường | Giá trị |
|---|---|
| **Tên project** | TTS Tiếng Cuối Chăm |
| **Mô tả nghiệp vụ** | Xây dựng hệ thống chuyển văn bản thành giọng nói (Text-to-Speech) cho ngôn ngữ thiểu số Cuối Chăm, dựa trên fine-tune mô hình VITS từ pretrained LJSpeech |
| **Use Case** | Tổng hợp giọng nói từ chuỗi âm vị (phoneme sequence) của tiếng Cuối Chăm|


---

## 2. NHÓM MODEL

| Trường | Giá trị |
|---|---|
| **Model sử dụng** | VITS (Variational Inference with adversarial learning for end-to-end TTS) – framework Coqui TTS |
| **Kiến trúc** | End-to-end TTS: Text Encoder → Flow-based Decoder → HiFi-GAN Vocoder, có Duration Predictor và KL Divergence |
| **Chiến lược** | Fine-tune từ pretrained (`tts_models--en--ljspeech--vits`), reinit text encoder do vocab khác hoàn toàn |
| **Pretrained gốc** | `/home/anhht/.local/share/tts/tts_models--en--ljspeech--vits/model_file.pth` |

### Sơ đồ pipeline

```
Phoneme Input (space-separated)
        ↓
CuoiChamPhonemesWithLabeling (tokenizer tùy chỉnh)
        ↓
VITS Text Encoder (reinit)
        ↓
Normalizing Flow + Duration Predictor
        ↓
HiFi-GAN Decoder
        ↓
Waveform Output (.wav)
```

---

## 3. NHÓM DATA

### 3.1 Dataset

| Trường | Giá trị |
|---|---|
| **Tên dataset** | `dataset` (dataset chính); `cuoi_cham_tts_dataset_for_debug` (thử nghiệm, debug nhanh với các tập data được chia nhỏ) |
| **Data Location** | `/home/anhht/textToSpeechCuoiCham/dataset/`, `/home/anhht/textToSpeechCuoiCham/cuoi_cham_tts_dataset_for_debug/` |
| **Data Schema** | Định dạng LJSpeech: `metadata_train.csv` và `metadata_dev.csv`, mỗi dòng gồm `wav_id,phoneme_sequence,phonemized_sequence` |
| **Cấu trúc thư mục** | `dataset/wavs/` chứa file `.wav`; `metadata_train.csv` và `metadata_dev.csv` ở cùng cấp |

### 3.2 Cấu trúc âm tiết


Cấu trúc mỗi âm tiết: `[initial] [nucleus] [coda_marker] [tone]`

### 3.3 Preprocessing

| Script | Mô tả |
|---|---|
| `src/process_data.py` | Tiền xử lý dữ liệu thô, chuẩn hóa phoneme |
| `src/check_sample_rate.py` | Kiểm tra sample rate toàn bộ audio (yêu cầu 22050 Hz) |

**Quy tắc chuẩn hóa:**
- Dùng NFC Unicode normalization để tránh sai biệt diacritics
- Coda được chia đúng (`n_coda`, `ŋ_coda`, `w_coda`, v.v.) theo bài báo tiếng Cuối Chăm

---

## 4. NHÓM TRAINING

### 4.1 Cấu hình training

| Tham số | Giá trị |
|---|---|
| `batch_size` | 32 |
| `eval_batch_size` | 8 |
| `epochs` | 1000 |
| `lr` (main) | 1e-4 |
| `lr` (generator / discriminator) | 1e-5 |
| `lr_scheduler` | ExponentialLR, gamma = 0.999875 |
| `mixed_precision` | true |
| `num_loader_workers` | 4 |
| `save_step` | 10000 |
| `dur_warmup_steps` | 20000 |
| `kl_anneal_steps` | 50000 |

### 4.2 Loss weights

| Loss | Hệ số |
|---|---|
| `lambda_mel` | 45.0 |
| `lambda_fm` | 2.0 |
| `lambda_adv` | 1.0 |
| `lambda_dur` | 1.0 |
| `lambda_kl` | 1.0 |

### 4.3 Audio config

| Tham số | Giá trị |
|---|---|
| `sample_rate` | 22050 Hz |
| `fft_size` | 1024 |
| `win_length` | 1024 |
| `hop_length` | 256 |
| `num_mels` | 80 |
| `mel_fmin` | 0.0 |
| `mel_fmax` | null |

### 4.4 Characters config

| Tham số | Giá trị |
|---|---|
| `characters_class` | `TTS.tts.utils.text.characters.CuoiChamPhonemesWithLabeling` |
| `pad` | `<PAD>` |
| `eos` | `<EOS>` |
| `bos` | `<BOS>` |
| `blank` | `<BLNK>` |
| `add_blank` | true |
| `is_unique` | true |

### 4.5 Lệnh chạy training

```bash
cd /home/anhht/textToSpeechCuoiCham
source venv/bin/activate

# Train mới
python -m TTS.bin.train --config_path dataset/config.json

# Resume từ checkpoint
python -m TTS.bin.train --config_path dataset/config.json \
  --continue_path training_output/<run>/checkpoint_XXXX.pth
```

### 4.6 Output training

| Loại | Đường dẫn |
|---|---|
| Checkpoint  | `training_output/<run>/checkpoint_*.pth` |
| Best model | `training_output/<run>/best_model_*.pth` |
| Config đã lưu | `training_output/<run>/config.json` |
| Log trainer | `training_output/<run>/trainer_0_log.txt` |
| Tensorboard | `tensorboard --logdir training_output/<run>` |

---

## 5. NHÓM MODEL OUTPUT / EVALUATION

### 5.1 Checkpoint / Artifact

| Trường | Giá trị |
|---|---|
| **Checkpoint location** | `training_output/<run>/checkpoint_*.pth` |
| **Best model location** | `training_output/<run>/best_model_*.pth` |
| **Định dạng** | `.pth` (PyTorch) |

### 5.2 Evaluation Metrics

| Script | Metric |
|---|---|
| `src/cal_mcd_dtw.py` | MCD (Mel Cepstral Distortion) dùng DTW alignment |
| `src/analysis_output_wav.py` | Phân tích formant bằng Praat-parselmouth |

| Thư mục kết quả | Mô tả |
|---|---|
| `analysis_result/` | Output wav đánh giá theo checkpoint |
| `evaluation_results_N_th/` | Kết quả eval lần N |
| `evaluation_results_17th_with_coda/` | Kết quả eval lần N, có nhãn coda |
| `eval_mcd.csv` | File MCD tổng hợp |

---

## 6. NHÓM INFERENCE

### 6.1 Inference đơn lẻ

```bash
tts --model_path training_output/<run>/best_model_*.pth \
    --config_path training_output/<run>/config.json \
    --text "kʰr aː T¹" \
    --out_path out.wav
```

### 6.2 Inference theo batch

Tạo file `input.csv`, wav_id|phoneme_sequence|phonemized_sequence, sau đó dùng script:

```bash
python src/test_output.py
```

Hoặc chạy tập metadata_test.csv đã chia:

```bash
python src/inference_metadata_test.py
```

### 6.3 Inference pipeline flow

```
Chuỗi phoneme đầu vào (space-separated)
        ↓
CuoiChamPhonemesWithLabeling.tokenize()
        ↓
VITS forward pass (best_model_*.pth)
        ↓
Waveform 22050 Hz
        ↓
Lưu ra file .wav
```


## 7. NHÓM DEPLOYMENT

| Trường | Giá trị |
|---|---|
| **Runtime** | Python 3.x, virtualenv tại `venv/` |
| **Framework** | Coqui TTS |
| **GPU** | CUDA (xem log tensorboard để xác định GPU đã dùng) |
| **Yêu cầu** | Xem `requirements.txt` |
| **Cài đặt** | `pip install -r requirements.txt` trong virtualenv |

---

## 8. NHÓM Monitor

| Trường | Giá trị |
|---|---|
| **Logging** | `training_output/<run>/trainer_0_log.txt` |
| **Monitoring training** | Tensorboard: `tensorboard --logdir training_output/<run>` |
| **Kiểm tra pretrained** | `python checkPretrain.py` – kiểm tra embedding shape của model pretrained |

---


## 9. LỚP TOKENIZER TÙY CHỈNH – `CuoiChamPhonemesWithLabeling`

Lớp này thay thế tokenizer mặc định của Coqui TTS để phù hợp với cấu trúc âm vị tiếng Cuối Chăm.

### Đặc điểm kỹ thuật

| Thành phần | Mô tả |
|---|---|
| `PHONEME_INVENTORY` | Tập hợp tất cả token hợp lệ, sort giảm dần theo độ dài để ưu tiên khớp dài nhất |
| Tokenization | Theo khoảng trắng (space-based), phù hợp với dữ liệu có nhãn rõ |
| `normalize_phoneme()` | Chuẩn hóa NFC Unicode |
| `validate_text()` | Kiểm tra token hợp lệ trước khi đưa vào dataset |
| `is_unique=True` | Phát hiện token trùng lặp trong vocab |

### Nhóm âm vị

- `initials`: Phụ âm đầu
- `nuclei`: Nguyên âm 
- `codas`: Phụ âm cuối (dạng `x_coda`)
- `tones`: Thanh điệu (`T¹` đến `T⁶`)

### Tích hợp

Trong `config.json`:
```json
"characters_class": "TTS.tts.utils.text.characters.CuoiChamPhonemesWithLabeling"
```




# web
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

input: chuỗi phoneme gốc, model sẽ gọi tới class CuoiChamPhonemesWithLabeling để tách theo công thức cấu trúc âm vị tiếng Cuối Chăm.
ví dụ: tʌt⁷ˢ ʔʌl³


# Train
python -m TTS.bin.train --config_path dataset/config.json

