#check tần số lấy mẫu
import librosa
y, sr = librosa.load("./cuoi_cham_tts_dataset_phonemized/wavs/crdo-TOU_VOC1_W2.wav", sr=None)
print(f"Sample rate: {sr} Hz")
