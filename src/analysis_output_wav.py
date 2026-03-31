"""
Alternative: Sử dụng Praat-parselmouth để phân tích formant
Không phụ thuộc vào matplotlib visualization
"""

import parselmouth
from parselmouth.praat import call
import pandas as pd
import numpy as np
from pathlib import Path


def extract_formants_detailed(audio_file, max_formant=5500):
    """
    Extract formant values chi tiết từ audio
    
    Args:
        audio_file: path to .wav file
        max_formant: 5000 cho giọng nam, 5500 cho giọng nữ
    
    Returns:
        DataFrame with formant values
    """
    print(f"🔍 Analyzing: {audio_file}")
    
    sound = parselmouth.Sound(audio_file)
    
    # Extract formant object
    formant = call(sound, "To Formant (burg)", 
                   0.01,           # time step
                   5.0,            # max number of formants
                   max_formant,    # maximum formant Hz
                   0.025,          # window length
                   50)             # pre-emphasis from Hz
    
    # Extract all formant values
    results = []
    for frame_idx in range(formant.n_frames):
        time = formant.x1 + frame_idx * formant.dx
        
        f1 = call(formant, "Get value at time", 1, time, 'Hertz', 'Linear')
        f2 = call(formant, "Get value at time", 2, time, 'Hertz', 'Linear')
        f3 = call(formant, "Get value at time", 3, time, 'Hertz', 'Linear')
        
        if f1 and f2 and f3:  # Only valid frames
            results.append({
                'time': time,
                'F1': f1,
                'F2': f2,
                'F3': f3
            })
    
    df = pd.DataFrame(results)
    print(f"  ✓ Extracted {len(df)} formant frames")
    print(f"  Duration: {df['time'].max():.2f}s")
    
    return df


def compare_formants_detailed(ai_file, real_file, output_dir='./formant_analysis'):
    """So sánh formant chi tiết giữa AI và Real"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract formants
    print("\n" + "="*60)
    print("EXTRACTING FORMANTS")
    print("="*60)
    
    ai_formants = extract_formants_detailed(ai_file)
    real_formants = extract_formants_detailed(real_file)
    
    # Save to CSV
    ai_csv = Path(output_dir) / 'ai_formants.csv'
    real_csv = Path(output_dir) / 'real_formants.csv'
    
    ai_formants.to_csv(ai_csv, index=False)
    real_formants.to_csv(real_csv, index=False)
    
    print(f"\n✓ Saved formant data:")
    print(f"  AI:   {ai_csv}")
    print(f"  Real: {real_csv}")
    
    # Statistical comparison
    print("\n" + "="*60)
    print("FORMANT STATISTICS COMPARISON")
    print("="*60)
    
    for formant_name in ['F1', 'F2', 'F3']:
        print(f"\n{formant_name}:")
        print(f"  AI:")
        print(f"    Mean: {ai_formants[formant_name].mean():.1f} Hz")
        print(f"    Std:  {ai_formants[formant_name].std():.1f} Hz")
        print(f"    Min:  {ai_formants[formant_name].min():.1f} Hz")
        print(f"    Max:  {ai_formants[formant_name].max():.1f} Hz")
        
        print(f"  Real:")
        print(f"    Mean: {real_formants[formant_name].mean():.1f} Hz")
        print(f"    Std:  {real_formants[formant_name].std():.1f} Hz")
        print(f"    Min:  {real_formants[formant_name].min():.1f} Hz")
        print(f"    Max:  {real_formants[formant_name].max():.1f} Hz")
        
        diff = abs(ai_formants[formant_name].mean() - real_formants[formant_name].mean())
        print(f"  → Mean difference: {diff:.1f} Hz")
        
        if diff > 100:
            print(f"  ⚠️  WARNING: Large deviation detected!")
    
    # Frame-by-frame comparison
    analyze_frame_differences(ai_formants, real_formants, output_dir)
    
    # Vowel space analysis
    analyze_vowel_space(ai_formants, real_formants, output_dir)
    
    return ai_formants, real_formants


def analyze_frame_differences(ai_df, real_df, output_dir):
    """Phân tích sự khác biệt theo từng frame"""
    
    print("\n" + "="*60)
    print("FRAME-BY-FRAME ANALYSIS")
    print("="*60)
    
    # Align dataframes by time (interpolate if needed)
    min_len = min(len(ai_df), len(real_df))
    ai_sample = ai_df.iloc[:min_len].reset_index(drop=True)
    real_sample = real_df.iloc[:min_len].reset_index(drop=True)
    
    # Calculate differences
    diff_df = pd.DataFrame({
        'time_ai': ai_sample['time'],
        'time_real': real_sample['time'],
        'F1_diff': abs(ai_sample['F1'] - real_sample['F1']),
        'F2_diff': abs(ai_sample['F2'] - real_sample['F2']),
        'F3_diff': abs(ai_sample['F3'] - real_sample['F3'])
    })
    
    # Find problematic frames
    threshold_f1 = 100  # Hz
    threshold_f2 = 150  # Hz
    
    problem_frames = diff_df[
        (diff_df['F1_diff'] > threshold_f1) | 
        (diff_df['F2_diff'] > threshold_f2)
    ]
    
    print(f"\nProblematic frames (F1>{threshold_f1}Hz or F2>{threshold_f2}Hz diff):")
    print(f"  Total: {len(problem_frames)}/{len(diff_df)} frames ({len(problem_frames)/len(diff_df)*100:.1f}%)")
    
    if len(problem_frames) > 0:
        print(f"\n  Top 10 worst frames:")
        worst = problem_frames.nlargest(10, 'F1_diff')
        for idx, row in worst.iterrows():
            print(f"    Time {row['time_ai']:.2f}s: F1 diff={row['F1_diff']:.0f}Hz, F2 diff={row['F2_diff']:.0f}Hz")
    
    # Save diff data
    diff_csv = Path(output_dir) / 'formant_differences.csv'
    diff_df.to_csv(diff_csv, index=False)
    print(f"\n✓ Saved difference data: {diff_csv}")


def analyze_vowel_space(ai_df, real_df, output_dir):
    """Phân tích vowel space (F1-F2 plot)"""
    
    print("\n" + "="*60)
    print("VOWEL SPACE ANALYSIS")
    print("="*60)
    
    # Calculate convex hull area (approximation of vowel space size)
    from scipy.spatial import ConvexHull
    
    try:
        ai_points = np.column_stack([ai_df['F2'], ai_df['F1']])
        real_points = np.column_stack([real_df['F2'], real_df['F1']])
        
        ai_hull = ConvexHull(ai_points)
        real_hull = ConvexHull(real_points)
        
        print(f"\nVowel space area (convex hull):")
        print(f"  AI:   {ai_hull.volume:.0f} Hz²")
        print(f"  Real: {real_hull.volume:.0f} Hz²")
        print(f"  Ratio: {ai_hull.volume/real_hull.volume:.2f}")
        
        if ai_hull.volume < real_hull.volume * 0.8:
            print(f"  ⚠️  AI vowel space is significantly smaller!")
            print(f"     → Nguyên âm có thể bị phát âm không rõ ràng")
        
    except Exception as e:
        print(f"  Cannot compute convex hull: {e}")
    
    # Save vowel space data
    vowel_space_data = {
        'source': ['AI'] * len(ai_df) + ['Real'] * len(real_df),
        'F1': list(ai_df['F1']) + list(real_df['F1']),
        'F2': list(ai_df['F2']) + list(real_df['F2'])
    }
    vowel_csv = Path(output_dir) / 'vowel_space.csv'
    pd.DataFrame(vowel_space_data).to_csv(vowel_csv, index=False)
    print(f"\n✓ Saved vowel space data: {vowel_csv}")
    print(f"  → Import vào Excel/Tableau để visualize F1 vs F2 scatter plot")


def generate_praat_report(ai_file, real_file, output_dir):
    """Generate comprehensive Praat analysis report"""
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    ai_formants, real_formants = compare_formants_detailed(ai_file, real_file, output_dir)
    
    # Generate text report
    report_path = Path(output_dir) / 'formant_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("VITS FORMANT ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"AI File:   {ai_file}\n")
        f.write(f"Real File: {real_file}\n\n")
        
        f.write("FORMANT STATISTICS:\n")
        f.write("-"*60 + "\n\n")
        
        for formant in ['F1', 'F2', 'F3']:
            f.write(f"{formant}:\n")
            f.write(f"  AI:   {ai_formants[formant].mean():.1f} ± {ai_formants[formant].std():.1f} Hz\n")
            f.write(f"  Real: {real_formants[formant].mean():.1f} ± {real_formants[formant].std():.1f} Hz\n")
            f.write(f"  Diff: {abs(ai_formants[formant].mean() - real_formants[formant].mean()):.1f} Hz\n\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION GUIDE:\n")
        f.write("="*60 + "\n\n")
        f.write("""
F1 (First Formant):
- Typical range: 200-1000 Hz
- Correlates with tongue height (high vs low vowels)
- Large deviation → wrong vowel height
  Example: /i/ (F1~300Hz) vs /a/ (F1~700Hz)

F2 (Second Formant):
- Typical range: 800-3000 Hz
- Correlates with tongue frontness (front vs back vowels)
- Large deviation → wrong vowel backness
  Example: /i/ (F2~2200Hz) vs /u/ (F2~800Hz)

F3 (Third Formant):
- Typical range: 2000-3500 Hz
- Important for consonant distinction
- Helps identify retroflex vs non-retroflex sounds

COMMON ISSUES:
1. F1/F2 both shifted → Completely wrong vowel
2. F1 correct, F2 wrong → Vowel backness error
3. Formants "blurred" → Coarticulation issues
4. Sudden jumps → Concatenation/alignment errors
        """)
    
    print(f"\n✓ Report saved: {report_path}")
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  1. {output_dir}/ai_formants.csv")
    print(f"  2. {output_dir}/real_formants.csv")
    print(f"  3. {output_dir}/formant_differences.csv")
    print(f"  4. {output_dir}/vowel_space.csv")
    print(f"  5. {output_dir}/formant_analysis_report.txt")
    print("\nNext steps:")
    print("  - Import CSVs into Excel/Tableau for visualization")
    print("  - Use Praat GUI to listen to problematic time segments")
    print("  - Check training data for underrepresented phonemes")


if __name__ == '__main__':
    # Install: pip install praat-parselmouth scipy pandas
    
    generate_praat_report(
        ai_file='./evaluation_results_18th_with_coda/crdo-TOU_VOC2_W212_AI.wav',
        real_file='./evaluation_results_18th_with_coda/crdo-TOU_VOC2_W212_GOC.wav',
        output_dir='./formant_analysis'
    )