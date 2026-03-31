#!/usr/bin/env python3
"""
MCD Evaluation sử dụng thư viện mel-cepstral-distance
Approach: Library-based cho reliability và development speed
"""

import os
import glob
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

# Import thư viện MCD
try:
    from mel_cepstral_distance import compare_audio_files
    from mel_cepstral_distance.utils import audio_preprocessing
    HAS_MCD_LIB = True
    print("✅ mel-cepstral-distance library đã được load thành công")
except ImportError:
    HAS_MCD_LIB = False
    print("❌ mel-cepstral-distance library chưa được cài đặt")
    print("📦 Cài đặt: pip install mel-cepstral-distance")
    print("📖 Docs: https://github.com/DigitalPhonetics/mel-cepstral-distance")

def preprocess_audio_files(ai_path, gt_path, target_sr=22050):
    """
    Preprocess audio files trước khi tính MCD
    
    Args:
        ai_path: Path đến file AI generated
        gt_path: Path đến file ground truth
        target_sr: Target sample rate
    
    Returns:
        tuple: (preprocessed_ai_path, preprocessed_gt_path, success)
    """
    try:
        # Load audio files
        y_ai, sr_ai = sf.read(ai_path)
        y_gt, sr_gt = sf.read(gt_path)
        
        # Convert stereo to mono
        if y_ai.ndim > 1:
            y_ai = np.mean(y_ai, axis=1)
        if y_gt.ndim > 1:
            y_gt = np.mean(y_gt, axis=1)
        
        # Normalize audio levels
        y_ai = y_ai / np.max(np.abs(y_ai)) * 0.9
        y_gt = y_gt / np.max(np.abs(y_gt)) * 0.9
        
        # Resample nếu cần
        if sr_ai != target_sr:
            y_ai = librosa.resample(y_ai, orig_sr=sr_ai, target_sr=target_sr)
        if sr_gt != target_sr:
            y_gt = librosa.resample(y_gt, orig_sr=sr_gt, target_sr=target_sr)
        
        # Tạo temp files cho thư viện (nếu cần)
        temp_dir = Path("temp_mcd")
        temp_dir.mkdir(exist_ok=True)
        
        temp_ai_path = temp_dir / f"temp_ai_{Path(ai_path).stem}.wav"
        temp_gt_path = temp_dir / f"temp_gt_{Path(gt_path).stem}.wav"
        
        # Save processed files
        sf.write(temp_ai_path, y_ai, target_sr)
        sf.write(temp_gt_path, y_gt, target_sr)
        
        return str(temp_ai_path), str(temp_gt_path), True
        
    except Exception as e:
        print(f"⚠️ Preprocessing error: {e}")
        return ai_path, gt_path, False

def calculate_mcd_with_library(ai_path, gt_path, use_dtw=True, preprocess=True):
    """
    Tính MCD sử dụng mel-cepstral-distance library
    
    Args:
        ai_path: Path đến file AI
        gt_path: Path đến file ground truth  
        use_dtw: Sử dụng DTW alignment
        preprocess: Có preprocess audio không
    
    Returns:
        dict: Kết quả MCD với metadata
    """
    if not HAS_MCD_LIB:
        raise ImportError("mel-cepstral-distance library chưa được cài đặt")
    
    try:
        # Preprocess nếu cần
        if preprocess:
            proc_ai_path, proc_gt_path, success = preprocess_audio_files(ai_path, gt_path)
            if not success:
                print(f"⚠️ Using original files without preprocessing")
                proc_ai_path, proc_gt_path = ai_path, gt_path
        else:
            proc_ai_path, proc_gt_path = ai_path, gt_path
        
        # Tính MCD với thư viện
        # Library tự động handle DTW, MCEP extraction, etc.
        mcd_value, penalty = compare_audio_files(
            reference_file=proc_gt_path,
            synthesized_file=proc_ai_path,
            use_dtw=use_dtw
        )
        
        # Cleanup temp files nếu có
        if preprocess and proc_ai_path != ai_path:
            try:
                os.remove(proc_ai_path)
                os.remove(proc_gt_path)
            except:
                pass
        
        return {
            'mcd': mcd_value,
            'penalty': penalty,
            'total_cost': mcd_value + penalty,
            'use_dtw': use_dtw,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'mcd': np.nan,
            'penalty': np.nan, 
            'total_cost': np.nan,
            'use_dtw': use_dtw,
            'status': f'error: {str(e)}'
        }

def batch_evaluate_mcd(eval_dir, use_dtw=True, preprocess=True, max_mcd_threshold=15.0):
    """
    Batch evaluation MCD cho nhiều file pairs
    
    Args:
        eval_dir: Thư mục chứa files
        use_dtw: Sử dụng DTW alignment
        preprocess: Preprocess audio files
        max_mcd_threshold: Threshold để filter outliers
    
    Returns:
        dict: Kết quả evaluation với statistics
    """
    if not HAS_MCD_LIB:
        print("❌ Cannot proceed without mel-cepstral-distance library")
        return {}
    
    print(f"🎵 Bắt đầu MCD evaluation trong: {eval_dir}")
    print(f"⚙️ DTW: {use_dtw}, Preprocess: {preprocess}, Threshold: {max_mcd_threshold}")
    print("="*60)
    
    # Tìm tất cả AI files
    ai_files = sorted(glob.glob(os.path.join(eval_dir, "*_AI.wav")))
    
    if not ai_files:
        print(f"❌ Không tìm thấy file *_AI.wav trong {eval_dir}")
        return {}
    
    results = []
    successful_evaluations = 0
    
    for ai_path in ai_files:
        basename = os.path.basename(ai_path).replace("_AI.wav", "")
        gt_path = os.path.join(eval_dir, basename + "_GOC.wav")
        
        print(f"🔄 Processing: {basename}")
        
        # Check ground truth file exists
        if not os.path.exists(gt_path):
            print(f"  ❌ Missing ground truth: {basename}_GOC.wav")
            results.append({
                'basename': basename,
                'mcd': np.nan,
                'penalty': np.nan,
                'status': 'missing_gt'
            })
            continue
        
        # Calculate MCD
        result = calculate_mcd_with_library(
            ai_path=ai_path,
            gt_path=gt_path, 
            use_dtw=use_dtw,
            preprocess=preprocess
        )
        
        # Add basename to result
        result['basename'] = basename
        result['ai_path'] = ai_path
        result['gt_path'] = gt_path
        
        # Print result
        if result['status'] == 'success':
            dtw_text = "DTW" if use_dtw else "No-DTW"
            print(f"  ✅ MCD ({dtw_text}): {result['mcd']:.3f} dB, Penalty: {result['penalty']:.3f}")
            
            # Check threshold
            if result['mcd'] <= max_mcd_threshold:
                successful_evaluations += 1
            else:
                print(f"     ⚠️ MCD > {max_mcd_threshold} dB (outlier)")
        else:
            print(f"  ❌ Error: {result['status']}")
        
        results.append(result)
    
    print("="*60)
    
    # Calculate statistics
    valid_results = [r for r in results if r['status'] == 'success' and not np.isnan(r['mcd'])]
    mcd_values = [r['mcd'] for r in valid_results]
    
    # Filter by threshold
    filtered_values = [val for val in mcd_values if val <= max_mcd_threshold]
    
    statistics = {
        'total_files': len(ai_files),
        'successful_evaluations': len(valid_results),
        'files_within_threshold': len(filtered_values),
        'threshold': max_mcd_threshold
    }
    
    if filtered_values:
        statistics.update({
            'mean_mcd': np.mean(filtered_values),
            'std_mcd': np.std(filtered_values),
            'min_mcd': np.min(filtered_values),
            'max_mcd': np.max(filtered_values),
            'median_mcd': np.median(filtered_values)
        })
        
        # Quality assessment
        mean_mcd = statistics['mean_mcd']
        if mean_mcd < 4.0:
            quality_level = "Xuất sắc (Excellent)"
        elif mean_mcd < 6.0:
            quality_level = "Tốt (Good)"
        elif mean_mcd < 8.0:
            quality_level = "Khá (Fair)"
        elif mean_mcd < 10.0:
            quality_level = "Trung bình (Average)"
        else:
            quality_level = "Cần cải thiện (Needs Improvement)"
        
        statistics['quality_assessment'] = quality_level
    
    return {
        'results': results,
        'statistics': statistics,
        'settings': {
            'use_dtw': use_dtw,
            'preprocess': preprocess,
            'eval_dir': eval_dir
        }
    }

def print_evaluation_report(evaluation_data):
    """
    In báo cáo kết quả evaluation
    """
    if not evaluation_data:
        print("❌ Không có dữ liệu để báo cáo")
        return
    
    stats = evaluation_data['statistics']
    settings = evaluation_data['settings']
    
    print(f"\n📊 MCD EVALUATION REPORT")
    print("="*60)
    print(f"📁 Directory: {settings['eval_dir']}")
    print(f"⚙️ Settings: DTW={settings['use_dtw']}, Preprocess={settings['preprocess']}")
    print(f"📈 Files processed: {stats['successful_evaluations']}/{stats['total_files']}")
    print(f"✅ Within threshold (<= {stats['threshold']} dB): {stats['files_within_threshold']}")
    
    if 'mean_mcd' in stats:
        print(f"\n📊 STATISTICS (filtered):")
        print(f"   Mean MCD: {stats['mean_mcd']:.3f} ± {stats['std_mcd']:.3f} dB")
        print(f"   Median MCD: {stats['median_mcd']:.3f} dB") 
        print(f"   Range: {stats['min_mcd']:.3f} - {stats['max_mcd']:.3f} dB")
        print(f"   Quality: {stats['quality_assessment']}")
    
    print("="*60)

def compare_dtw_vs_no_dtw(eval_dir, preprocess=True):
    """
    So sánh kết quả MCD với và không DTW
    """
    print(f"\n🔄 COMPARING DTW vs No-DTW")
    print("="*60)
    
    # Evaluate với DTW
    print("1️⃣ Evaluating with DTW...")
    dtw_results = batch_evaluate_mcd(eval_dir, use_dtw=True, preprocess=preprocess)
    
    print("\n2️⃣ Evaluating without DTW...")
    no_dtw_results = batch_evaluate_mcd(eval_dir, use_dtw=False, preprocess=preprocess)
    
    # Compare results
    print(f"\n📊 COMPARISON SUMMARY:")
    
    if dtw_results and no_dtw_results:
        dtw_stats = dtw_results['statistics']
        no_dtw_stats = no_dtw_results['statistics']
        
        if 'mean_mcd' in dtw_stats and 'mean_mcd' in no_dtw_stats:
            print(f"   DTW Mean MCD: {dtw_stats['mean_mcd']:.3f} dB")
            print(f"   No-DTW Mean MCD: {no_dtw_stats['mean_mcd']:.3f} dB") 
            print(f"   Difference: {abs(dtw_stats['mean_mcd'] - no_dtw_stats['mean_mcd']):.3f} dB")
            
            if dtw_stats['mean_mcd'] < no_dtw_stats['mean_mcd']:
                print("   🏆 DTW gives better (lower) MCD scores")
            else:
                print("   🏆 No-DTW gives better (lower) MCD scores")
    
    return dtw_results, no_dtw_results

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Configuration
    EVAL_DIR = "evaluation_results_17th_with_coda"
    USE_DTW = True
    PREPROCESS = True
    THRESHOLD = 15.0
    
    # Check if directory exists
    if not os.path.exists(EVAL_DIR):
        print(f"❌ Thư mục {EVAL_DIR} không tồn tại!")
        print("📁 Tạo thư mục và đặt các file *_AI.wav và *_GOC.wav vào đó")
        exit(1)
    
    # Check library installation
    if not HAS_MCD_LIB:
        print("\n🚀 QUICK START GUIDE:")
        print("1. pip install mel-cepstral-distance")
        print("2. Chạy lại script này")
        exit(1)
    
    # Main evaluation
    print(f"🎯 MCD Evaluation sử dụng mel-cepstral-distance library")
    
    # Single evaluation
    evaluation_results = batch_evaluate_mcd(
        eval_dir=EVAL_DIR,
        use_dtw=USE_DTW, 
        preprocess=PREPROCESS,
        max_mcd_threshold=THRESHOLD
    )
    
    # Print report
    print_evaluation_report(evaluation_results)
    
    # Optional: Compare DTW vs No-DTW
    compare_approaches = input("\n🤔 So sánh DTW vs No-DTW? (y/n): ").lower() == 'y'
    if compare_approaches:
        compare_dtw_vs_no_dtw(EVAL_DIR, preprocess=PREPROCESS)
    
    # Cleanup temp directory
    temp_dir = Path("temp_mcd")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files")
    
    print(f"\n✅ MCD Evaluation hoàn thành!")