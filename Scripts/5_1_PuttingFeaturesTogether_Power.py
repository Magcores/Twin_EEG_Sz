# %% Feature Aggregation with ROI and Global Averages
import os
import pandas as pd
import numpy as np
from glob import glob

# === Paths ===
BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
feature_dir = os.path.join(BASE_DIR, "Preprocessed", "PowerAndFoooF")
output_dir = os.path.join(BASE_DIR, "Preprocessed", "Features")
os.makedirs(output_dir, exist_ok=True)

# === Frequency bands ===
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),
}

# === Simplified ROI Mapping ===
roi_map = {
    'Prefrontal': ['Fp', 'AF'],
    'Frontal': ['F'],
    'Central': ['C', 'FC', 'CP'],
    'Parietal': ['P', 'PO'],
    'Temporal': ['T', 'TP', 'FT'],
    'Occipital': ['O', 'Iz']
}

def assign_roi(channel):
    for roi, prefixes in roi_map.items():
        for prefix in prefixes:
            if prefix in channel:
                return roi
    return 'Unknown'

# === Load and process all subject feature files ===
all_files = glob(os.path.join(feature_dir, '*_features.csv'))
all_channel_rows = []
all_roi_rows = []

for file in all_files:
    df = pd.read_csv(file)
    subject_id = df['subject_id'].iloc[0]

    # Add ROI labels
    df['roi'] = df['channel'].apply(assign_roi)

    # For each band, extract the peak with the highest PW in that band range per channel
    band_features = []
    for (subj, chan), group in df.groupby(['subject_id', 'channel']):
        base_row = group.iloc[0][['subject_id', 'channel', 'aperiodic_offset', 'aperiodic_exponent', 'r_squared', 'fit_error',
                                  'delta_lin','delta_dB','theta_lin','theta_dB','alpha_lin','alpha_dB','beta_lin','beta_dB','gamma_lin','gamma_dB']]
        row = base_row.to_dict()

        for band, (low, high) in bands.items():
            band_group = group[(group['CF'] >= low) & (group['CF'] <= high)]
            if not band_group.empty:
                best_peak = band_group.loc[band_group['PW'].idxmax()]
                row[f'CF_{band}'] = best_peak['CF']
                row[f'PW_{band}'] = best_peak['PW']
                row[f'BW_{band}'] = best_peak['BW']
            else:
                row[f'CF_{band}'] = np.nan
                row[f'PW_{band}'] = np.nan
                row[f'BW_{band}'] = np.nan

        row['roi'] = assign_roi(row['channel'])
        band_features.append(row)

    subj_df = pd.DataFrame(band_features)
    all_channel_rows.append(subj_df)

    # ROI Averages
    roi_df = subj_df.groupby('roi').mean(numeric_only=True).reset_index()
    roi_df.insert(0, 'subject_id', subject_id)

    # Global Average (all channels)
    global_vals = subj_df.mean(numeric_only=True)
    global_row = pd.DataFrame([global_vals])
    global_row.insert(0, 'roi', 'Global')
    global_row.insert(0, 'subject_id', subject_id)

    all_roi_rows.append(pd.concat([roi_df, global_row], ignore_index=True))

# === Save final outputs ===
all_channels_df = pd.concat(all_channel_rows, ignore_index=True)
all_rois_df = pd.concat(all_roi_rows, ignore_index=True)

all_channels_df.to_csv(os.path.join(output_dir, 'fooof_AllChannels.csv'), index=False)
all_rois_df.to_csv(os.path.join(output_dir, 'fooof_AllROIs.csv'), index=False)

print("✅ Aggregation complete. Files saved:")
print(" - fooof_AllChannels.csv")
print(" - fooof_AllROIs.csv")

# % Combine to subject-level wide format

# Pivot channel data
channel_wide = (
    all_channels_df
    .drop(columns=['roi'])
    .pivot(index='subject_id', columns='channel')
)
channel_wide.columns = [f"{metric}_{chan}" for metric, chan in channel_wide.columns]

# Pivot ROI data
roi_wide = (
    all_rois_df
    .pivot(index='subject_id', columns='roi')
)
roi_wide.columns = [f"{metric}_{roi}" for metric, roi in roi_wide.columns if metric != 'subject_id']

# Merge both wide tables
final_df = pd.concat([channel_wide, roi_wide], axis=1).reset_index()

# Save
final_df.to_csv(os.path.join(output_dir, 'fooof_FinalWide_perSubject.csv'), index=False)
print("🧾 Final subject-level wide table saved as fooof_wide_perSubject.csv")


# %%
