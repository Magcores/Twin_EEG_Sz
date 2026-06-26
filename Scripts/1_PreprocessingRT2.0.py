"""
Final preprocessing script for resting-state EEG (Twins SZ project).
- Processes one or multiple subjects
- Saves preprocessed raw and epoched data
- Logs detailed QC metrics and saves plots
- Stores consistently bad channels and EEG metadata for each subject
- Summarizes all QC into one master table

Author: Martin Antunez
"""

# %% Imports
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import AutoReject
import os
import pandas as pd
import json
from glob import glob

# %% Setup: define mode and paths
process_all_subjects = True  # 🔁 Set to True to process all subjects in the Raws folder
base_dir = r'P:\YOUR_DATA_PATH_HERE'
raws_dir = os.path.join(base_dir, 'Raws')
preprocessed_dir = os.path.join(base_dir, 'Preprocessed')
os.makedirs(preprocessed_dir, exist_ok=True)

# Discover subject files
raw_files = glob(os.path.join(raws_dir, '*_raw.fif'))
all_subjects = [os.path.basename(f).split('_')[0] for f in raw_files]

if process_all_subjects:
    # Filter out already processed subjects
    subjects = []
    for sid in all_subjects:
        out_file = os.path.join(preprocessed_dir, 'Continuous', f'{sid}_preprocessed_raw.fif')
        if not os.path.exists(out_file):
            subjects.append(sid)
    if not subjects:
        print("All subjects already processed. Nothing to do.")
else:
    # Manually define subject for testing
    subjects = ['1DZAHCXXXXXXX']  # 🔧 Change this ID for testing different subjects

# Master QC summary
qc_summary = []

for subject_id in subjects:
    out_raw_file = os.path.join(preprocessed_dir, 'Continuous', f'{subject_id}_preprocessed_raw.fif')
    if os.path.exists(out_raw_file):
        print(f"Skipping {subject_id}, already processed.")
        continue

    print(f"--- Processing subject: {subject_id} ---")
    paths = {
        'raw_file': os.path.join(raws_dir, f'{subject_id}_raw.fif'),
        'out_raw': os.path.join(preprocessed_dir, 'Continuous', f'{subject_id}_preprocessed_raw.fif'),
        'out_epochs': os.path.join(preprocessed_dir, 'Epoched', f'{subject_id}_epoched-epo.fif'),
        'plot_dir': os.path.join(preprocessed_dir, 'Plots', subject_id),
        'bad_channels_table': os.path.join(preprocessed_dir, 'Plots', subject_id, f'{subject_id}_bad_channels_table.csv'),
        'metadata_json': os.path.join(preprocessed_dir, 'Plots', subject_id, f'{subject_id}_metadata_summary.json')
    }
    os.makedirs(paths['plot_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(paths['out_raw']), exist_ok=True)
    os.makedirs(os.path.dirname(paths['out_epochs']), exist_ok=True)

    # Load raw data
    raw = mne.io.read_raw_fif(paths['raw_file'], preload=True)
    raw.plot(n_channels=64, duration=10, scalings='auto', title='Raw EEG', show=False).savefig(
        os.path.join(paths['plot_dir'], f'{subject_id}_raw_overview.png'))

    raw.resample(256, npad="auto")
    raw.set_montage('standard_1020')
    raw.notch_filter(freqs=50)
    raw.filter(1., 100., fir_design='firwin')

    # PSD-based bad channel detection
    psd = raw.copy().compute_psd(fmin=1., fmax=100., n_fft=1024, n_overlap=512)
    psds = psd.get_data()
    freqs = psd.freqs
    psds_db = 10 * np.log10(psds)
    mean_band_power = psds_db[:, (freqs >= 1) & (freqs <= 45)].mean(axis=1)
    z_scores = zscore(mean_band_power)
    bad_channels = [raw.ch_names[i] for i, z in enumerate(z_scores) if np.abs(z) > 2.5]
    raw.info['bads'] = bad_channels
    plt.figure()
    for i, ch_psd in enumerate(psds_db):
        color = 'r' if raw.ch_names[i] in bad_channels else 'k'
        plt.plot(freqs, ch_psd, color=color, alpha=0.5)
    plt.title("PSD (1–100 Hz), Bad Channels Highlighted")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.xlim([1, 100])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(paths['plot_dir'], f'{subject_id}_psd_bad_channels.png'))
    plt.close()

    raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference('average', projection=False)

    # ICA + ICLabel
    ica = ICA(n_components=0.999, method='infomax', fit_params=dict(extended=True), random_state=97)
    ica.fit(raw)
    labels = label_components(raw, ica, method='iclabel')
    ica_labels = labels['labels']
    brain_idx = [i for i, label in enumerate(ica_labels) if label == 'brain']
    ica.exclude = [i for i in range(len(ica_labels)) if i not in brain_idx]
    ica.apply(raw)

    if ica.exclude:
        ica.plot_components(picks=ica.exclude, show=False).savefig(
            os.path.join(paths['plot_dir'], f'{subject_id}_ica_excluded.png'))

    # AutoReject
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    ar = AutoReject(n_interpolate=[1, 2, 4], consensus=[0.1, 0.2, 0.3])
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)
    epochs_clean = ar.transform(epochs)
    fig = reject_log.plot(show=False)
    fig.savefig(os.path.join(paths['plot_dir'], f'{subject_id}_autoreject_log.png'))
    plt.close(fig)

    # Save raw and epochs
    raw.save(paths['out_raw'], overwrite=True)
    epochs_clean.save(paths['out_epochs'], overwrite=True)

    # QC summary
    n_epochs = len(epochs)
    n_rejected = reject_log.bad_epochs.sum()
    channel_counts = reject_log.labels.astype(bool).sum(axis=0)
    channel_names = epochs.ch_names
    most_corrected = sorted(zip(channel_names, channel_counts), key=lambda x: -x[1])
    consistently_bad = [ch for ch, count in most_corrected if count > 0.9 * n_epochs]

    raw.info['temp'] = raw.info.get('temp', {})
    raw.info['temp']['epochwise_consistently_bad'] = consistently_bad

    pd.DataFrame({
        'Channel': consistently_bad,
        'InterpolatedEpochs': [dict(most_corrected)[ch] for ch in consistently_bad],
        'TotalEpochs': n_epochs,
        'PercentInterpolated': [100 * dict(most_corrected)[ch] / n_epochs for ch in consistently_bad]
    }).to_csv(paths['bad_channels_table'], index=False)

    # Metadata summary
    meta_summary = {
        'subject_id': subject_id,
        'n_channels': raw.info['nchan'],
        'sfreq': raw.info['sfreq'],
        'highpass': raw.info['highpass'],
        'lowpass': raw.info['lowpass'],
        'duration_sec': raw.times[-1] if raw.times.size else None,
        'meas_date': str(raw.info['meas_date']) if raw.info['meas_date'] else None,
        'bad_channels': raw.info['bads'],
        'epochwise_consistently_bad': consistently_bad,
        'n_epochs': n_epochs,
        'n_rejected': int(n_rejected),
        'n_cleaned': int(n_epochs - n_rejected)
    }
    with open(paths['metadata_json'], 'w') as f_json:
        json.dump(meta_summary, f_json, indent=4)
    qc_summary.append(meta_summary)

# Append to group QC summary with timestamp
from datetime import datetime
qc_df = pd.DataFrame(qc_summary)
qc_df['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

qc_file = os.path.join(preprocessed_dir, 'Plots', 'qc_summary_all_subjects.csv')
if os.path.exists(qc_file):
    existing_qc = pd.read_csv(qc_file)
    qc_df = pd.concat([existing_qc, qc_df], ignore_index=True)
    qc_df = qc_df.drop_duplicates(subset=['subject_id'], keep='last')
qc_df.to_csv(qc_file, index=False)
print("\nFinished all processing.")
