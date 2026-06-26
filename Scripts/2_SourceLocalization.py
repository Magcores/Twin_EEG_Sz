import os
import time
import mne
import pandas as pd
from glob import glob
from datetime import datetime
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
import matplotlib.pyplot as plt

# Optional visualization backend warnings
missing_backends = []
try:
    import pyvistaqt
except ImportError:
    missing_backends.append("pyvistaqt")
try:
    import ipyevents
    import ipywidgets
except ImportError:
    missing_backends.append("ipywidgets/ipyevents")

if missing_backends:
    print("\n⚠️  Optional visualization backends are missing.")
    print("   You can still run the script, but 3D brain visualizations will be skipped.")
    print("🔍 To enable them, run the following in your terminal:")
    if "pyvistaqt" in missing_backends:
        print("   pip install pyvistaqt qtpy")
    if "ipywidgets/ipyevents" in missing_backends:
        print("   pip install ipywidgets ipyevents")
    print("   Or with conda:")
    print("   conda install -c conda-forge pyvistaqt qtpy ipywidgets ipyevents\n")

# ============================ SETTINGS ============================
force_rerun = False
inverse_method = 'dSPM'
snr = 3.0
lambda2 = 1.0 / snr ** 2
resample_rate = 100  # Optional resample rate to reduce memory usage (Original 256 Hz)

base_dir = r'P:\YOUR_DATA_PATH_HERE'
preprocessed_dir = os.path.join(base_dir, 'Preprocessed')
source_dir = os.path.join(preprocessed_dir, 'Source')
raw_dir = os.path.join(preprocessed_dir, 'Continuous')
qc_file = os.path.join(source_dir, 'source_summary_all_subjects.csv')

subjects_dir = os.path.join(source_dir, 'fs_templates')
os.environ['SUBJECTS_DIR'] = subjects_dir

os.makedirs(source_dir, exist_ok=True)
os.makedirs(subjects_dir, exist_ok=True)

# Ensure fsaverage template is available
fsaverage_subject_path = os.path.join(subjects_dir, 'fsaverage')
if not os.path.exists(fsaverage_subject_path):
    print("📥 Downloading fsaverage (template head model)...")
    fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)

# Check required fsaverage files
required_paths = [
    os.path.join(fsaverage_subject_path, 'surf', 'lh.white'),
    os.path.join(fsaverage_subject_path, 'bem', 'fsaverage-inner_skull-bem.fif')
]
for req_path in required_paths:
    if not os.path.exists(req_path):
        print(f"❌ Missing required fsaverage file: {req_path}")
        exit(1)

# Find subjects to process
raw_files = glob(os.path.join(raw_dir, '*_preprocessed_raw.fif'))
all_subject_ids = [os.path.basename(f).split('_')[0] for f in raw_files]

subject_ids = []
for sid in all_subject_ids:
    subj_out_dir = os.path.join(source_dir, sid)
    out_file = os.path.join(subj_out_dir, f'{sid}_source_{inverse_method}')
    summary_file = os.path.join(subj_out_dir, 'source_summary.csv')
    if force_rerun or not (os.path.exists(out_file + '-lh.stc') and os.path.exists(summary_file)):
        subject_ids.append(sid)

if not subject_ids:
    print("✅ All subjects already have source estimates.")
else:
    print(f"🔍 Found {len(subject_ids)} subjects to process.\n")

# ============================ MAIN LOOP ============================

for idx, subject_id in enumerate(subject_ids, start=1):
    print(f"\n🧠 [{idx}/{len(subject_ids)}] Starting subject: {subject_id}")
    t0_total = time.time()

    try:
        print("📥 Step: Loading preprocessed raw data...")
        raw = mne.io.read_raw_fif(os.path.join(raw_dir, f'{subject_id}_preprocessed_raw.fif'), preload=True)

        # Re-reference EEG to average (required for inverse modeling)
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()

        # OPTIONAL: Downsample to reduce memory usage
        if resample_rate is not None and resample_rate < raw.info['sfreq']:
            print(f"\n⚠️  WARNING: Resampling from {raw.info['sfreq']} Hz to {resample_rate} Hz to reduce memory usage.")
            print("   This reduces temporal resolution and may affect analyses like connectivity or PAC.\n")
            raw.resample(resample_rate)

        print("🧠 Step: Setting up source space and BEM...")
        trans = os.path.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-trans.fif')
        src = mne.setup_source_space('fsaverage', spacing='oct6', add_dist=False, subjects_dir=subjects_dir)
        bem = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir)
        bem_sol = mne.make_bem_solution(bem)

        print("➡️ Step: Computing forward model...")
        fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, eeg=True, mindist=5.0, n_jobs=1)

        print("📊 Step: Estimating noise covariance...")
        noise_cov = mne.compute_raw_covariance(raw, method='empirical')

        print(f"🔁 Step: Applying inverse method ({inverse_method})...")
        inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        stc = apply_inverse_raw(raw, inverse_operator, lambda2=lambda2, method=inverse_method, verbose=False)

        if stc is None or stc.data.size == 0 or stc.times.size == 0:
            raise ValueError("STC output is invalid or empty.")
        else:
            print("✅ STC file passed integrity checks: data and time dimensions are valid.")

        print("💾 Step: Saving STC and summary data...")
        subj_out_dir = os.path.join(source_dir, subject_id)
        os.makedirs(subj_out_dir, exist_ok=True)
        stc_path = os.path.join(subj_out_dir, f'{subject_id}_source_{inverse_method}')
        stc.save(stc_path)

        mean_activity = stc.data.mean(axis=0)
        df_mean = pd.DataFrame({'time': stc.times, 'mean_activity': mean_activity})
        df_mean.to_csv(os.path.join(subj_out_dir, 'source_summary.csv'), index=False)

        print("📊 Extracting ROI time courses using 'aparc' parcellation...")
        labels = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=subjects_dir)

        # Remove 'unknown' and labels with no vertices in the source space
        labels = [label for label in labels if 'unknown' not in label.name.lower()]
        src_vertices = src[0]['vertno'].tolist() + src[1]['vertno'].tolist()
        labels = [label for label in labels if any(v in src_vertices for v in label.vertices)]

        if not labels:
            raise RuntimeError("No valid labels with vertices found in the source space.")

        print(f"✅ Using {len(labels)} labels for time course extraction:")
        for label in labels:
            print(f" - {label.name}")

        label_ts = mne.extract_label_time_course(stc, labels, src, mode='mean', return_generator=False)
        df_roi = pd.DataFrame(label_ts.T, columns=[label.name for label in labels])
        df_roi.insert(0, 'time', stc.times)
        df_roi.to_csv(os.path.join(subj_out_dir, f'{subject_id}_roi_timecourses.csv'), index=False)

        print("✅ Finished subject:", subject_id)

    except Exception as e:
        print(f"❌ Error processing {subject_id}: {e}")
