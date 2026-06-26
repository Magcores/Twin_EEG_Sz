# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, hilbert
from glob import glob
from itertools import combinations
from tqdm import tqdm

# ========================== SETTINGS ==========================

BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
source_dir = os.path.join(BASE_DIR, r"Preprocessed\Source")
output_dir = os.path.join(BASE_DIR, r"Preprocessed\Connectivity_wPLI_bands")
os.makedirs(output_dir, exist_ok=True)

sfreq = 100  # Sampling frequency in Hz

# Define frequency bands
frequency_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

# ========================== FUNCTIONS ==========================

def bandpass_filter(data, sfreq, low, high, order=4):
    nyquist = 0.5 * sfreq
    low /= nyquist
    high /= nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_wpli(sig1, sig2):
    x1 = hilbert(sig1)
    x2 = hilbert(sig2)
    im_part = np.imag(x1 * np.conj(x2))

    num = np.sum(np.abs(im_part) * np.sign(im_part))
    den = np.sum(np.abs(im_part))
    wpli = num / den if den != 0 else 0
    return max(0.0, min(1.0, wpli))

def plot_heatmap(matrix, labels, title, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap='viridis', square=True)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ========================== MAIN LOOP ==========================

roi_files = sorted(glob(os.path.join(source_dir, '*', '*_roi_timecourses.csv')))
all_matrices = {band: [] for band in frequency_bands.keys()}
all_subjects = []

for file_path in tqdm(roi_files, desc="Computing wPLI"):
    subject_id = os.path.basename(file_path).split('_')[0]
    subject_folder = os.path.join(output_dir, subject_id)
    if os.path.exists(subject_folder):
        print(f"✅ Skipping {subject_id}, already processed.")
        continue

    print(f"🔄 Processing subject: {subject_id}")

    try:
        df = pd.read_csv(file_path)
        roi_names = df.columns[1:]
        raw_data = df[roi_names].values.T

        os.makedirs(subject_folder, exist_ok=True)

        for band_name, (f_low, f_high) in frequency_bands.items():
            print(f"    ▶ Band: {band_name}")

            filtered_data = np.array([
                bandpass_filter(ts, sfreq, f_low, f_high) for ts in raw_data
            ])

            n_rois = len(roi_names)
            wpli_matrix = np.zeros((n_rois, n_rois))

            for i, j in combinations(range(n_rois), 2):
                wpli = compute_wpli(filtered_data[i], filtered_data[j])
                wpli_matrix[i, j] = wpli
                wpli_matrix[j, i] = wpli

            np.fill_diagonal(wpli_matrix, 0)

            df_wpli = pd.DataFrame(wpli_matrix, index=roi_names, columns=roi_names)
            out_csv = os.path.join(subject_folder, f'wpli_{band_name}.csv')
            df_wpli.to_csv(out_csv)

            out_plot = os.path.join(subject_folder, f'wpli_{band_name}_heatmap.png')
            plot_heatmap(wpli_matrix, roi_names, f'{subject_id} - {band_name} band', out_plot)

            all_matrices[band_name].append(wpli_matrix)

        all_subjects.append(subject_id)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")


# ========================== GROUP AVERAGES ==========================
print("📂 Starting group average computation")
print("WARNING: DONT FORGET RUNNING SETTINGS AGAIN BEFORE THIS CHUNK")

group_source = output_dir
subject_folders = [f for f in os.listdir(group_source) if os.path.isdir(os.path.join(group_source, f))]

band_matrices = {band: [] for band in frequency_bands}
roi_names = None

for subject_id in subject_folders:
    for band_name in frequency_bands:
        file_path = os.path.join(group_source, subject_id, f'wpli_{band_name}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            if roi_names is None:
                roi_names = df.columns
            band_matrices[band_name].append(df.values)

for band_name, matrices in band_matrices.items():
    if matrices:
        print(f"📊 Computing group average for: {band_name}")
        mean_matrix = np.mean(matrices, axis=0)
        df_mean = pd.DataFrame(mean_matrix, index=roi_names, columns=roi_names)

        mean_csv = os.path.join(output_dir, f'group_mean_wpli_{band_name}.csv')
        df_mean.to_csv(mean_csv)

        mean_plot = os.path.join(output_dir, f'group_mean_wpli_{band_name}_heatmap.png')
        plot_heatmap(mean_matrix, roi_names, f'Group Average - {band_name} band', mean_plot)

print("✅ Group averages completed.")
# %%
