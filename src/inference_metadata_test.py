""" 
--model_path          : Đường dẫn đến file checkpoint của model (.pth).
  --config_path         : Đường dẫn đến file config.json tương ứng của model.
  --test_metadata_file  : Đường dẫn đến file metadata (.csv) chứa các mẫu cần test.
                          File này phải có format: path_to_wavs|text|text_normalized
  --original_wavs_dir   : textToSpeechCuoiCham/dataset/wavs
  --output_eval_dir     : Tên thư mục để lưu kết quả so sánh.

  lệnh chạy:
  python inference_metadata_test.py

"""
import os
import pandas as pd
import subprocess
import shutil



MODEL_PATH = "./best_model/checkpoint1220000.pth"
CONFIG_PATH = "./config.json"

TEST_METADATA_FILE = "./dataset/metadata_test.csv"

ORIGINAL_WAVS_DIR = "./dataset/wavs"

OUTPUT_EVAL_DIR = "evaluation_results_18th_with_coda"


# Tạo thư mục output nếu chưa có
if not os.path.exists(OUTPUT_EVAL_DIR):
    os.makedirs(OUTPUT_EVAL_DIR)

# Đọc file metadata của tập test
try:
    df_test = pd.read_csv(TEST_METADATA_FILE, sep='|', header=None, names=['basename', 'text', 'text_normalized'])
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file metadata test: {TEST_METADATA_FILE}")
    exit()

print(f"Tìm thấy {len(df_test)} mẫu trong tập test. Bắt đầu tạo file so sánh...")

# Vòng lặp qua từng mẫu trong tập test
for index, row in df_test.iterrows():
    basename = row['basename']
    text_to_speak = row['text']
    
    print(f"\n--- Đang xử lý: {basename} ---")
    print(f"Văn bản: {text_to_speak}")

    # --- Tạo file của AI ---
    output_ai_filename = f"{basename}_AI.wav"
    output_ai_filepath = os.path.join(OUTPUT_EVAL_DIR, output_ai_filename)
    
    command = [
        "tts",
        "--model_path", MODEL_PATH,
        "--config_path", CONFIG_PATH,
        "--text", text_to_speak,
        "--out_path", output_ai_filepath
    ]
    
    try:
        print("Đang gọi model AI để tạo giọng nói...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"  > Đã tạo file AI: {output_ai_filename}")
    except subprocess.CalledProcessError as e:
        print(f"  LỖI KHI TẠO FILE AI: {e.stderr}")
        continue # Bỏ qua mẫu này nếu có lỗi

    # --- Sao chép file GỐC ---
    original_wav_filename = f"{basename}.wav"
    original_wav_filepath = os.path.join(ORIGINAL_WAVS_DIR, original_wav_filename)
    
    output_original_filename = f"{basename}_GOC.wav"
    output_original_filepath = os.path.join(OUTPUT_EVAL_DIR, output_original_filename)

    if os.path.exists(original_wav_filepath):
        print("Đang sao chép file gốc...")
        shutil.copy(original_wav_filepath, output_original_filepath)
        print(f"  > Đã sao chép file gốc: {output_original_filename}")
    else:
        print(f"  CẢNH BÁO: Không tìm thấy file gốc tại {original_wav_filepath}")

print(f"\n--- HOÀN THÀNH ---")
print(f"Toàn bộ file so sánh đã được tạo trong thư mục: {OUTPUT_EVAL_DIR}")