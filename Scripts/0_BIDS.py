# Convert to .edf
# Create BIDS structure

# Install dependencies (run in terminal, not in script)
# pip install mne mne-bids
# pip install edfio  # optional, not required for standard EDF export

# %%
import os
from mne.io import read_raw_fif
from mne.export import export_raw
from mne_bids import write_raw_bids, BIDSPath
import mne

from mne.io import read_raw_edf


# Define paths
BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
fif_dir = os.path.join(BASE_DIR, "Raws")
edf_dir = os.path.join(BASE_DIR, "Raw_edf")
bids_root = os.path.join(BASE_DIR, "BIDS")

# Create output directories if needed
os.makedirs(edf_dir, exist_ok=True)
os.makedirs(bids_root, exist_ok=True)

# %%
print("Starting Loop 1: Convert .fif to .edf")

converted_fif = 0
skipped_fif = 0

for file_name in os.listdir(fif_dir):
    if file_name.endswith('.fif'):
        fif_path = os.path.join(fif_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        edf_path = os.path.join(edf_dir, f"{base_name}.edf")

        if os.path.exists(edf_path):
            print(f"[SKIP] {file_name} already converted to EDF.")
            skipped_fif += 1
            continue

        print(f"[CONVERT] Converting {file_name} to EDF...")
        raw = read_raw_fif(fif_path, preload=True)
        export_raw(edf_path, raw, fmt='edf')
        print(f"[DONE] Saved EDF: {edf_path}")
        converted_fif += 1

print(f"Loop 1 complete: {converted_fif} converted, {skipped_fif} skipped.")
print("="*60)

# %%

print("Starting Loop 2: Convert .edf to BIDS")

converted_bids = 0
skipped_bids = 0

for file_name in os.listdir(edf_dir):
    if file_name.endswith('.edf'):
        edf_path = os.path.join(edf_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        subject_id = base_name.replace('_raw', '')

        # Create expected BIDS folder structure
        subject_folder = os.path.join(bids_root, f"sub-{subject_id}", "ses-01", "eeg")
        os.makedirs(subject_folder, exist_ok=True)

        # Expected output BIDS .edf file path
        bids_edf_path = os.path.join(
            subject_folder,
            f"sub-{subject_id}_ses-01_task-rest_eeg.edf"
        )

        # Skip if BIDS EDF already exists
        if os.path.exists(bids_edf_path):
            print(f"[SKIP] {file_name} already in BIDS format.")
            skipped_bids += 1
            continue

        print(f"[BIDS] Converting {file_name} to BIDS...")

        # Load EDF
        raw_edf = read_raw_edf(edf_path, preload=True)

        # Create BIDS path object
        bids_path = BIDSPath(subject=subject_id, session="01", task="rest", root=bids_root)

        # Write to BIDS
        write_raw_bids(raw_edf, bids_path, overwrite=True, allow_preload=True, format='EDF')
        print(f"[DONE] BIDS written to: {bids_edf_path}")
        converted_bids += 1

print(f"Loop 2 complete: {converted_bids} converted, {skipped_bids} skipped.")
print("=" * 60)
print("All processing finished.")



# %%

#Include participants.tsv with metadata to BIDS

import os
import pandas as pd
# pip install --user xlrd # For reading the old Excel

# Path to the Excel metadata
excel_path = r"P:\YOUR_EXCEL_PATH_HERE\VIP_MASTER_selected_variables_270325.xls"
edf_dir = os.path.join(BASE_DIR, "Raw_edf")
bids_root = os.path.join(BASE_DIR, "BIDS")

# Read Excel metadata
df_metadata = pd.read_excel(excel_path)

# Extract subject ID from the first 7 characters
df_metadata['subject_id'] = df_metadata['ID'].astype(str).str[:7].str.upper()

# Extract pair_id from the last 7 digits of the full ID
df_metadata['pair_id'] = df_metadata['ID'].astype(str).str[-7:]
# Prepend age as string to the pair_id
df_metadata['pair_id'] = df_metadata['age'].astype(str) + "" + df_metadata['pair_id']


# Get subject IDs from EDF files
edf_subjects = sorted({
    os.path.splitext(f)[0].replace('_raw', '').zfill(6).upper()
    for f in os.listdir(edf_dir)
    if f.endswith('.edf')
})

# Clean and pad edf_subjects to ensure 7-character IDs
edf_subjects = [str(s).strip().upper() for s in edf_subjects]
edf_subjects = [s if len(s) == 7 else '0' + s if len(s) == 6 else s for s in edf_subjects]

# Filter to only include metadata for those subjects
df_participants = df_metadata[df_metadata['subject_id'].isin(edf_subjects)].copy()


# Rename and clean columns
df_participants = df_participants.rename(columns={
    'group': 'group',
    'twinab': 'twin_type_ab',
    'Pr0/HP1/HC2': 'clinical_group (Pr0/HP1/HC2)',
    'gender (0=female, 1=male)': 'sex',
    'age': 'age'
})

# Convert gender codes to BIDS format
df_participants['sex'] = df_participants['sex'].map({0: 'F', 1: 'M'}).fillna('n/a')

# Create BIDS participant_id
df_participants['participant_id'] = 'sub-' + df_participants['subject_id']

# Reorder columns with pair_id before twin_type_ab
cols = ['participant_id', 'sex', 'age', 'group', 'pair_id', 'twin_type_ab', 'clinical_group (Pr0/HP1/HC2)']
df_participants = df_participants[cols]

# Save to participants.tsv
os.makedirs(bids_root, exist_ok=True)
participants_tsv_path = os.path.join(bids_root, 'participants.tsv')
df_participants.to_csv(participants_tsv_path, sep='\t', index=False)

print(f"[INFO] participants.tsv saved with {len(df_participants)} participants.")


# %%
