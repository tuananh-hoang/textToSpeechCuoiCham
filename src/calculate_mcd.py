import os
import glob
import librosa
import numpy as np
import soundfile as sf

try:
    import pyworld as pw
    import pysptk
    HAS_PYWORLD = True
except Exception:
    HAS_PYWORLD = False

def extract_mcep(y, sr, order=24, frame_period=5.0):
    """
    Trích xuất Mel-Cepstral coefficients
    
    Args:
        y: Audio signal
        sr: Sample rate
        order: Số coefficients (thường 24 cho speech)
        frame_period: Frame period trong ms
    
    Returns:
        mcep: Mel-cepstral coefficients [frames, coefficients]
    """
    if HAS_PYWORLD:
        # WORLD vocoder - method chính xác cho speech analysis
        y = y.astype(np.double)
        _f0, t = pw.harvest(y, sr, frame_period=frame_period)
        sp = pw.cheaptrick(y, _f0, t, sr)
        
        # Alpha parameter cho mel-scale warping
        alpha = 0.42 if sr >= 16000 else 0.31
        mcep = pysptk.sp2mc(sp, order=order, alpha=alpha)
    else:
        # Fallback: MFCC (ít chính xác hơn)
        mcep = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=order).T
    
    return mcep

def calculate_mcd(mcep_ref, mcep_syn, use_dtw=True):
    """
    Tính MCD với hoặc không DTW alignment
    
    Args:
        mcep_ref: Reference MCEP [frames, coefficients]
        mcep_syn: Synthesized MCEP [frames, coefficients] 
        use_dtw: Có sử dụng DTW alignment không
    
    Returns:
        mcd_value: MCD score in dB
    """
    if use_dtw:
        # DTW alignment để xử lý timing mismatch
        # Transpose để librosa.dtw nhận [features, frames]
        D, wp = librosa.sequence.dtw(
            X=mcep_ref.T, 
            Y=mcep_syn.T, 
            metric="euclidean"
        )
        
        # Extract aligned sequences từ warping path
        # wp là list of (i,j) tuples - không cần reverse!
        mcep_ref_aligned = mcep_ref[[i for i, j in wp]]
        mcep_syn_aligned = mcep_syn[[j for i, j in wp]]
    else:
        # Simple truncation alignment
        L = min(len(mcep_ref), len(mcep_syn))
        mcep_ref_aligned = mcep_ref[:L]
        mcep_syn_aligned = mcep_syn[:L]
    
    # MCD formula: 10/ln(10) * sqrt(2) * sqrt(mean(sum((c1-c2)^2)))
    # Bỏ qua c0 coefficient (thường là energy/power)
    diff = mcep_ref_aligned[:, 1:] - mcep_syn_aligned[:, 1:]
    
    # Tính per-frame Euclidean distance
    per_frame_dist = np.sqrt(np.sum(diff ** 2, axis=1))
    
    # MCD final calculation
    mcd = (10.0 / np.log(10)) * np.sqrt(2.0) * np.mean(per_frame_dist)
    
    return mcd

def evaluate_mcd_batch(eval_dir, sr=22050, use_dtw=True):
    """
    Batch evaluation MCD cho nhiều file
    """
    ai_files = sorted(glob.glob(os.path.join(eval_dir, "*_AI.wav")))
    results = []
    
    print(f"Tìm thấy {len(ai_files)} file AI để đánh giá...")
    
    for ai_path in ai_files:
        basename = os.path.basename(ai_path).replace("_AI.wav", "")
        gt_path = os.path.join(eval_dir, basename + "_GOC.wav")
        
        if not os.path.exists(gt_path):
            print(f"[CẢNH BÁO] Không có file gốc cho {basename}")
            continue
        
        try:
            # Load và preprocess audio
            y_ai, sr_ai = sf.read(ai_path)
            y_gt, sr_gt = sf.read(gt_path)
            
            # Convert stereo to mono nếu cần
            if y_ai.ndim > 1:
                y_ai = np.mean(y_ai, axis=1)
            if y_gt.ndim > 1:
                y_gt = np.mean(y_gt, axis=1)
            
            # Resample về target sr nếu cần
            if sr_ai != sr:
                y_ai = librosa.resample(y_ai, orig_sr=sr_ai, target_sr=sr)
            if sr_gt != sr:
                y_gt = librosa.resample(y_gt, orig_sr=sr_gt, target_sr=sr)
            
            # Extract MCEP features
            mcep_ai = extract_mcep(y_ai, sr)
            mcep_gt = extract_mcep(y_gt, sr)
            
            # Validate MCEP shapes
            if mcep_ai.shape[1] != mcep_gt.shape[1]:
                print(f"[LỖI] {basename}: MCEP dimension mismatch")
                continue
            
            # Compute MCD
            mcd_val = calculate_mcd(mcep_gt, mcep_ai, use_dtw=use_dtw)
            results.append((basename, mcd_val))
            
            dtw_status = "với DTW" if use_dtw else "không DTW"
            print(f"{basename}: MCD ({dtw_status}) = {mcd_val:.3f} dB")
            
        except Exception as e:
            print(f"[LỖI] {basename}: {e}")
    
    return results

def print_statistics(results, threshold=15.0):
    """
    In thống kê kết quả MCD
    """
    if not results:
        print("Không có kết quả nào để thống kê.")
        return
    
    # Filter outliers
    mcd_values = [r[1] for r in results]
    filtered_values = [val for val in mcd_values if val <= threshold]
    
    print(f"\n{'='*50}")
    print(f"THỐNG KÊ MCD RESULTS")
    print(f"{'='*50}")
    print(f"Tổng số file: {len(results)}")
    print(f"File hợp lệ (<= {threshold} dB): {len(filtered_values)}")
    
    if filtered_values:
        mean_mcd = np.mean(filtered_values)
        std_mcd = np.std(filtered_values)
        min_mcd = np.min(filtered_values)
        max_mcd = np.max(filtered_values)
        
        print(f"Mean MCD: {mean_mcd:.3f} ± {std_mcd:.3f} dB")
        print(f"Min MCD: {min_mcd:.3f} dB")
        print(f"Max MCD: {max_mcd:.3f} dB")
        
        # Chất lượng categories
        if mean_mcd < 4.0:
            quality = "Xuất sắc"
        elif mean_mcd < 6.0:
            quality = "Tốt"
        elif mean_mcd < 8.0:
            quality = "Khá"
        else:
            quality = "Cần cải thiện"
        
        print(f"Đánh giá chất lượng: {quality}")
    else:
        print("Không có file nào pass threshold!")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    EVAL_DIR = "evaluation_results_11th"
    SR = 22050
    
    if not os.path.exists(EVAL_DIR):
        print(f"Thư mục {EVAL_DIR} không tồn tại!")
        exit(1)
    
    print("Bắt đầu đánh giá MCD với DTW alignment...")
    results = evaluate_mcd_batch(EVAL_DIR, sr=SR, use_dtw=True)
    print_statistics(results)
    
    # So sánh với không DTW
    print(f"\n{'-'*50}")
    print("So sánh: MCD không DTW...")
    results_no_dtw = evaluate_mcd_batch(EVAL_DIR, sr=SR, use_dtw=False)
    print_statistics(results_no_dtw)