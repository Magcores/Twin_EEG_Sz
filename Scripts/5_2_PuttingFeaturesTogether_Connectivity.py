# %%
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

# ================== SETTINGS ==================
BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
input_dir = os.path.join(BASE_DIR, r'Preprocessed\Connectivity_wPLI_bands')
output_dir = os.path.join(BASE_DIR, r'Preprocessed\Features')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'all_subjects_wpli_longformat.csv')

frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# ================== LONG FORMAT BUILD ==================
all_rows = []

subject_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for subject_id in tqdm(subject_folders, desc="Processing subjects"):
    for band in frequency_bands:
        file_path = os.path.join(input_dir, subject_id, f'wpli_{band}.csv')
        if not os.path.exists(file_path):
            continue
        
        try:
            df = pd.read_csv(file_path, index_col=0)
            for roi1 in df.index:
                for roi2 in df.columns:
                    wpli = df.at[roi1, roi2]
                    all_rows.append({
                        'subject_id': subject_id,
                        'frequency_band': band,
                        'roi_1': roi1,
                        'roi_2': roi2,
                        'wpli_value': wpli
                    })
        except Exception as e:
            print(f"❌ Failed on {subject_id} - {band}: {e}")

# Convert to DataFrame
long_df = pd.DataFrame(all_rows)

# Save to CSV
long_df.to_csv(output_file, index=False)
print(f"✅ Combined file saved to: {output_file}")


# %
import os
import pandas as pd
from tqdm import tqdm
from itertools import combinations

# ========= SETTINGS =========
input_dir = os.path.join(BASE_DIR, r'Preprocessed\Connectivity_wPLI_bands')
output_dir = os.path.join(BASE_DIR, r'Preprocessed\Features')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'wpli_features_wide.csv')

frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# ========= FEATURE TABLE BUILD =========
all_features = {}
subject_ids = []

subject_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for subject_id in tqdm(subject_folders, desc="Extracting features"):
    subject_feature_row = {}
    for band in frequency_bands:
        file_path = os.path.join(input_dir, subject_id, f'wpli_{band}.csv')
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, index_col=0)

        # Use only lower triangle (no self-loops or symmetric duplication)
        for i, roi1 in enumerate(df.index):
            for j in range(i):
                roi2 = df.columns[j]
                value = df.iat[i, j]
                feature_name = f'wpli_{band}_{roi1}_{roi2}'
                subject_feature_row[feature_name] = value

    if subject_feature_row:
        all_features[subject_id] = subject_feature_row
        subject_ids.append(subject_id)

# Convert to DataFrame
wide_df = pd.DataFrame.from_dict(all_features, orient='index')
wide_df.index.name = 'subject_id'
wide_df.reset_index(inplace=True)

# Save
wide_df.to_csv(output_file, index=False)
print(f"✅ Feature matrix saved: {output_file}")

# %%
