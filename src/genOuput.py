import torch
from TTS.api import TTS

# --- THAY ĐỔI CÁC THAM SỐ Ở ĐÂY ---

# 1. Đường dẫn đến file config và model của bạn
CONFIG_PATH = "cuoi_cham_tts_dataset_phonemized/config_sratch.json"
MODEL_PATH = "training_output/cuoi_cham_finetune_final_from_scratch_nhe_xindo-September-11-2025_10+29AM-f5a66b3/checkpoint_70000.pth"

# 2. Chuỗi phoneme bạn muốn test
INPUT_TEXT = "puːj⁴ kɒː⁵"

# 3. Tên file audio output
OUTPUT_PATH = "ket_qua_tu_script.wav"

# --- KẾT THÚC PHẦN THAY ĐỔI ---


# Kiểm tra xem có GPU không, nếu có thì dùng
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Sử dụng thiết bị: {device}")

# 1. Tải mô hình từ file config và checkpoint
print("Đang tải mô hình...")
try:
    tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH).to(device)
    print("✅ Tải mô hình thành công.")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    exit()

# 2. Lấy ra tokenizer từ model đã tải

tokenizer = tts.processor.tokenizer
# 3. GỌI HÀM TOKENIZE VÀ IN KẾT QUẢ RA MÀN HÌNH
print("-" * 50)
print(f"Input text: '{INPUT_TEXT}'")
try:
    # Gọi hàm .tokenize() mà bạn đã viết
    tokens = tokenizer.tokenize(INPUT_TEXT)
    print(f"===> KẾT QUẢ TOKENIZE: {tokens}")
except Exception as e:
    print(f"❌ Lỗi khi chạy tokenizer: {e}")
    exit()
print("-" * 50)

# 4. Chạy mô hình để tạo ra giọng nói
print(f"Đang tạo giọng nói và lưu vào file '{OUTPUT_PATH}'...")
try:
    # Sử dụng hàm tts_to_file của đối tượng tts cấp cao
    tts.tts_to_file(text=INPUT_TEXT, file_path=OUTPUT_PATH)
    print("🎉 Hoàn thành!")
except Exception as e:
    print(f"❌ Lỗi trong quá trình tạo giọng nói: {e}")