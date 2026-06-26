# %%
import os
import pandas as pd

# === SETTINGS ===
BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
features_dir = os.path.join(BASE_DIR, r'Preprocessed\Features')

wpli_file = os.path.join(features_dir, 'wpli_features_wide.csv')
fooof_file = os.path.join(features_dir, 'fooof_FinalWide_perSubject.csv')
output_file = os.path.join(features_dir, 'combined_features_wide.csv')
log_file = os.path.join(features_dir, 'combined_features_summary_log.txt')

# === LOAD FILES ===
df_wpli = pd.read_csv(wpli_file)
df_fooof = pd.read_csv(fooof_file)

# === MERGE ON 'subject_id' ===
merged_df = pd.merge(df_wpli, df_fooof, on='subject_id', how='inner')

# === SAVE MERGED FILE ===
merged_df.to_csv(output_file, index=False)

# === CREATE SUMMARY LOG ===
with open(log_file, 'w') as f:
    f.write("Combined Feature File Summary Log\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Combined file saved to:\n{output_file}\n\n")
    
    f.write("Dataset Overview:\n")
    f.write(f"- Number of subjects (rows): {merged_df.shape[0]}\n")
    f.write(f"- Number of features (columns): {merged_df.shape[1]}\n\n")

    f.write("Data Types:\n")
    dtype_counts = merged_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        f.write(f"  - {dtype}: {count} columns\n")
    f.write("\n")

    missing = merged_df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if not missing.empty:
        f.write("Columns with Missing Values (percentage):\n")
        for col, perc in missing.items():
            f.write(f"  - {col}: {perc:.2f}%\n")
    else:
        f.write("No missing values found.\n")

    f.write("\nMemory Usage:\n")
    mem_mb = merged_df.memory_usage(deep=True).sum() / 1024**2
    f.write(f"- Total memory usage: {mem_mb:.2f} MB\n")

print(f"Combined file and summary log saved:\n- Data: {output_file}\n- Log: {log_file}")

# %%
