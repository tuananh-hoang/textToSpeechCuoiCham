# Bàn giao dự án TTS Cuối Chăm (cập nhật)

## 1) Mục tiêu dự án
- Xây dựng hệ thống TTS phát âm "cuối Chăm" dựa trên mô hình VITS của Coqui TTS.
- Kết quả lưu checkpoint, best model, và audio inference.

## 2) Cấu trúc chính hiện tại

```
textToSpeechCuoiCham/
  ├─ dataset/              # dataset core (wavs + metadata (train/dev/test))
  ├─ cuoi_cham_tts_dataset_for_debug/  # thu nghiem voi cac tap data nho, debug data
  ├─ analysis_result/   # output evaluation (wavs) cua cac checkpoint
  ├─ best_model/   # checkpoint.pth và config.json
  ├─ training_output/        # output training (checkpoint/best/log)
  ├─ src/                   # inference, evaluation scripts
  │    ├─ analysis_output_wav.py  #Sử dụng Praat-parselmouth để phân tích formant
  │    ├─ cal_mcd_dtw.py     # MCD Evaluation
  │    ├─ genOutput.py        # sinh output với một chuỗi phoneme đầu vào cụ thể
  │    ├─ check_sample_rate.py # check sample rate của audio
  │    ├─ inference_metadata_test.py   # chạy test các phoneme theo batch
  ├─ app.py # web ui test kết quả
  ├─ requirements.txt
  ├─ checkPretrain.py  # check embedding shape của model pretrained
  └─ tai_lieu_ban_giao.md
```

## 3) Training

### 3.1 Cấu hình chính
- `config.json` (gốc) chứa:
  - `restore_path` pretrain từ coqui
  - `model: vits`
  - `audio.sample_rate`, `mel` config
  - `characters` (graphemes/phonemes)
  - `datasets`: path tới `Speaker_2/` hoặc dataset folder
  - `epochs`, `batch_size`, `lr`, loss weights

### 3.2 Chạy training hiện tại
- Dùng command tiêu chuẩn TTS:
```bash
source venv/bin/activate
python -m TTS.bin.train --config_path config.json
```
- Hoặc script `src/train_finetune.py` (khởi tạo Vits thủ công):
```bash
python src/train_finetune.py
```
- Với resume checkpoint:
```bash
python -m TTS.bin.train --config_path config.json --continue_path training_output/<run>/checkpoint_XXXX.pth
```

### 3.3 Kết quả training
- Checkpoints tại: `training_output/<run>/checkpoint_*.pth`
- Best model: `training_output/<run>/best_model_*.pth`
- Tensorboard: `tensorboard --logdir training_output/<run>`
- Log training: `/home/anhht/textToSpeechCuoiCham/training_output/<run>/trainer_0_log.txt`

## 5) Inference (synthesis)


1. Nếu dùng text phoneme:
```bash
tts --model_path training_output/<run>/best_model_*.pth --config_path training_output/<run>/config.json --text "kʌl..." --out_path out.wav
```
2. Dùng file list input (batch): tạo file `input.txt`, mỗi dòng text.

## 6) Evaluation
- Kết quả evaluation xuất ra `evaluation_results*/`.
- Lưu file MCD nếu có: `eval_mcd.csv`.
- Thử nghiệm nhiều version như `evaluation_results_10th/`, `..._17th_with_coda/`.

## 6.1) Class `CuoiChamPhonemesWithLabeling` (tùy chỉnh)
Lớp này được điều chỉnh để chia âm vị đúng theo bài báo tiếng Cuối Chăm:
- Sử dụng phân nhóm `initials`, `nuclei`, `codas`, `tones` để chuẩn hóa âm vị.
- Xây dựng `PHONEME_INVENTORY` bằng cách sort giảm dần độ dài.
- Tokenization theo khoảng trắng (space-based), phù hợp với dữ liệu có nhãn rõ ràng.

### 6.1.1) Cách tích hợp
1. Trong `config.json`, đặt `characters_class` thành class đã cài (`CuoiChamPhonemesWithLabeling`) hoặc đường dẫn module tương ứng.
2. Dữ liệu đầu vào phải là dạng phoneme space-separated (ví dụ `kʰr aː T² ...`).
3. Dùng `validate_text()` để kiểm tra token nằm trong vocab trước khi tạo dataset.
4. Nếu dùng data raw, chuyển qua preprocessing/phonemize để tạo định dạng space token trước.

### 6.1.2) Lưu ý kỹ thuật
- Class yêu cầu `is_unique=True` để phát hiện trùng lặp và đảm bảo vocab sạch.
- `normalize_phoneme()` dùng NFC để tránh sự khác biệt do diacritics.
- Nếu text có ký tự lạ, tokenization sẽ cảnh báo và bỏ qua.

## 7) Quick start ngắn gọn (1 phút)
```bash
cd /home/anhht/textToSpeechCuoiCham
source venv/bin/activate
# 1) Train model
python -m TTS.bin.train --config_path config.json
# 2) Synthesize
python -m TTS.bin.synthesize --model_path training_output/<run>/best_model_*.pth --config_path training_output/<run>/config.json --out_path out.wav --text "..."
```

---
