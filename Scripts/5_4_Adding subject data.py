# %%
import os
import pandas as pd

# File paths
BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
features_path = os.path.join(BASE_DIR, r'Preprocessed\Features\combined_features_wide.csv')
participants_path = os.path.join(BASE_DIR, r'BIDS\participants.tsv')
output_path = os.path.join(BASE_DIR, r'Preprocessed\Features\combined_features_wide_SubjectInfo.csv')

# Read data
features_df = pd.read_csv(features_path)
participants_df = pd.read_csv(participants_path, sep='\t')

# Clean participant IDs to match format in features_df
participants_df['subject_id'] = participants_df['participant_id'].str.replace('sub-', '', regex=False)

# ✅ Normalize subject_id in both DataFrames to 7-character strings (pad with '0' if 6)
features_df['subject_id'] = features_df['subject_id'].astype(str).str.strip().apply(
    lambda x: x.zfill(7)
)
participants_df['subject_id'] = participants_df['subject_id'].astype(str).str.strip().apply(
    lambda x: x.zfill(7)
)


# Get sets of IDs
features_ids = set(features_df['subject_id'].astype(str).str.strip())
participants_ids = set(participants_df['subject_id'].astype(str).str.strip())

# Matching and non-matching IDs
matching_ids = features_ids.intersection(participants_ids)
features_only = features_ids - participants_ids
participants_only = participants_ids - features_ids

print("✅ Matching subject_ids:", sorted(matching_ids))
print("❌ Present in features_df only (no match in participants.tsv):", sorted(features_only))
print("❌ Present in participants.tsv only (no match in features_df):", sorted(participants_only))

# Now merge
merged_df = pd.merge(features_df, participants_df, on='subject_id', how='left')

# Reorder columns: participant info first
participant_cols = [col for col in participants_df.columns if col != 'subject_id']
merged_df = merged_df[participant_cols + features_df.columns.tolist()]

# Save merged file
merged_df.to_csv(output_path, index=False)

# Get sets of IDs
features_ids = set(features_df['subject_id'])
participants_ids = set(participants_df['subject_id'])

# Matching and non-matching IDs
matching_ids = features_ids & participants_ids
features_only = features_ids - participants_ids
participants_only = participants_ids - features_ids

# Helper to get stats
def describe_ids(label, id_set):
    lengths = [len(x) for x in id_set]
    print(f"{label}: {len(id_set)} IDs")
    if lengths:
        print(f"  Lengths → Min: {min(lengths)}, Max: {max(lengths)}, Unique lengths: {set(lengths)}")

# Final output
describe_ids("✅ Matching IDs", matching_ids)
describe_ids("❌ In features_df only", features_only)
describe_ids("❌ In participants.tsv only", participants_only)
# %%
