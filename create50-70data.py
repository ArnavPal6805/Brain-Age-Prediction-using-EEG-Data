import pandas as pd
import numpy as np
import pywt
import os
import multiprocessing
from scipy.stats import entropy
from joblib import Parallel, delayed  # For parallel processing

def extract_age(file_path):
    """Extracts age from the first row of the CSV file."""
    with open(file_path, 'r') as f:
        age_line = f.readline().strip()  
        return int(age_line.split('=')[1])  # Extract integer age

def load_eeg_data(file_path, chunk_size=5000):
    """Loads EEG data efficiently using chunks."""
    return pd.concat(pd.read_csv(file_path, skiprows=2, chunksize=chunk_size), ignore_index=True)

def wavelet_energy_features(signal, wavelet='db4', level=3, keep_top=2):
    """Applies Wavelet Transform and keeps top energy coefficients."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energies = np.array([np.sum(np.abs(c) ** 2) for c in coeffs])  # Compute energy

    # Keep only the top `keep_top` highest-energy coefficients
    top_indices = np.argsort(energies)[-keep_top:]  
    selected_coeffs = [coeffs[i] for i in top_indices]

    # Vectorized feature extraction (Mean, Std, Energy, Entropy)
    features = np.hstack([
        [np.mean(c), np.std(c), np.sum(np.abs(c)), entropy(np.abs(c) + 1e-10)]
        for c in selected_coeffs
    ])
    
    return features

def process_segment(segment, wavelet='db4', level=3, keep_top=2):
    """Processes one EEG segment."""
    return np.hstack([wavelet_energy_features(segment[col], wavelet, level, keep_top) for col in segment.columns])

def segment_and_extract_features(df, segment_size=256, wavelet='db4', level=3, keep_top=2):
    """Splits EEG data into segments and extracts features using multiprocessing."""
    num_segments = len(df) // segment_size
    segments = (df.iloc[i * segment_size:(i + 1) * segment_size] for i in range(num_segments))

    # Parallel processing using multiple CPU cores
    features = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(process_segment)(segment, wavelet, level, keep_top) for segment in segments
    )

    return pd.DataFrame(features)

def preprocess_eeg(file_path, output_folder, wavelet='db4', level=3, keep_top=2, segment_size=256):
    """Processes a single EEG file and saves it in an age-based folder only if age is between 50 and 70."""

    # Extract Age
    age = extract_age(file_path)

    # Skip processing if age is not between 50 and 70
    if not (50 <= age <= 70):
        print(f"⏩ Skipped: {file_path} (Age: {age} not in [50, 70])")
        return

    # Load EEG Data efficiently
    df = load_eeg_data(file_path)

    # Apply Wavelet Transform and feature extraction
    feature_df = segment_and_extract_features(df, segment_size, wavelet, level, keep_top)

    # Convert to float32 for size efficiency
    feature_df = feature_df.astype(np.float32)

    # Create age-based subfolder
    age_folder = os.path.join(output_folder, str(age))
    os.makedirs(age_folder, exist_ok=True)

    # Save processed file
    file_name = os.path.basename(file_path).replace('.csv', '_processed.csv')
    output_file = os.path.join(age_folder, file_name)
    feature_df.to_csv(output_file, index=False)

    print(f"✅ Processed: {file_path} (Age: {age}) → Saved at: {output_file}")

def process_all_eeg_files(input_folder, output_folder, wavelet='db4', level=3, keep_top=2, segment_size=256):
    """Processes all EEG CSV files from an input folder using parallel processing."""
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' does not exist!")
        return

    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not files:
        print("⚠ No CSV files found in the input folder!")
        return

    # Parallel processing for multiple EEG files
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(preprocess_eeg)(os.path.join(input_folder, f), output_folder, wavelet, level, keep_top, segment_size)
        for f in files
    )

# Example usage
input_folder = "/kaggle/input/data-eeg-age-v1/data_eeg_age_v1/data2kaggle/train"
output_folder = "/kaggle/working/"

process_all_eeg_files(input_folder, output_folder)
