# %%
import os
import pandas as pd
import re

# ---------------------------------------------
# 📂 Load data
# ---------------------------------------------
BASE_DIR = r"P:\YOUR_DATA_PATH_HERE"  # TODO: Set this to your project data directory
file_path = os.path.join(BASE_DIR, r'Preprocessed\Features\combined_features_wide_SubjectInfo.csv')
save_path = os.path.join(BASE_DIR, "Outputs ML", "Only Connectivity")

df = pd.read_csv(file_path)
print(f"🔢 Original shape: {df.shape[0]} rows × {df.shape[1]} columns")

# -------------------------------
# -------------------------------
# Config starts here -------------------------------
# -------------------------------
# -------------------------------

# 🔧 CONFIG: PCA parameters
# You need to choose between EXPLAINED_VARIANCE_THRESHOLD or N_PCA_COMPONENTS
EXPLAINED_VARIANCE_THRESHOLD = 0.95  # Or set to None if using fixed number normally is 0.95
#EPLAINED_VARIANCE_THRESHOLD = None
#N_PCA_COMPONENTS = 30              # Set to 10, 20, etc. for fixed number
#N_PCA_COMPONENTS = 2 #2 Only for freq since PC1 is 97% of explained variance  
#N_PCA_COMPONENTS = None  
PCA_RANDOM_STATE = 42

# ---------------------------------------------
# 🔧 CONFIG: Define logic for which columns to KEEP
# ---------------------------------------------

import re

# Precompiled EEG channel pattern
eeg_channel_pattern = re.compile(
    r'_(AF|F|FC|C|CP|P|PO|O|Fp|FT|TP|T)[0-9]*[rlzt]?$|_(Fz|Cz|Pz|Oz|POz|Iz)$',
    re.IGNORECASE
)

# ROI label suffixes
roi_suffixes = [
    '_Frontal', '_Parietal', '_Occipital', '_Temporal',
    '_Global', '_Prefrontal', '_Central'
]

# Frequency bands
freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

def keep_column(col):
    col_lower = col.lower()

    # -------------------------------
    # ✅ GROUP 1: aperiodic + ROI only
    # -------------------------------
    if col.lower().startswith('aperiodic_'):
        if any(col.endswith(suffix) for suffix in roi_suffixes):
            return False  # ← Change to True if you want to keep

    # -------------------------------
    # ✅ GROUP 2: aperiodic + EEG channel only
    # -------------------------------
    if col.lower().startswith('aperiodic_'):
        if eeg_channel_pattern.search(col):
            return False  # ← Change to True if you want to keep

    # -------------------------------
    # ✅ GROUP 3: freq_band + _db_ + ROI only
    # -------------------------------
    if any(col_lower.startswith(f"{band}_") for band in freq_bands) and '_db_' in col_lower:
        if any(col.endswith(suffix) for suffix in roi_suffixes):
            return False  # ← Change to True if you want to keep

    # -------------------------------
    # ✅ GROUP 4: freq_band + _db_ + EEG channel only
    # -------------------------------
    if any(col_lower.startswith(f"{band}_") for band in freq_bands) and '_db_' in col_lower:
        if eeg_channel_pattern.search(col):
            return False  # ← Change to True if you want to keep

    # -------------------------------
    # ✅ GROUP 5: CF_, PW_, BW_ + ROI only
    # -------------------------------
    if any(col.startswith(prefix) for prefix in ['CF_', 'PW_', 'BW_']):
        if any(col.endswith(suffix) for suffix in roi_suffixes):
            return False  # ← Change to True if you want to keep

    # -------------------------------
    # ✅ GROUP 6: CF_, PW_, BW_ + EEG channel only
    # -------------------------------
    if any(col.startswith(prefix) for prefix in ['CF_', 'PW_', 'BW_']):
        if eeg_channel_pattern.search(col):
            return False  # ← Change to True if you want to keep

    # -------------------------------
    # ✅ GROUP 7: wpli_
    # -------------------------------
    if col.startswith('wpli_'):
        return True  # ← Change to True if you want to keep

    # 🔒 Otherwise, exclude
    return False


# -------------------------------
# -------------------------------
# Config ends here -------------------------------
# -------------------------------
# -------------------------------

# Automatically include essential subject metadata if present
always_keep = [
    'participant_id', 'age', 'sex', 'group', 'pair_id',
    'twin_type_ab', 'clinical_group (Pr0/HP1/HC2)', 'subject_id'
]

# ---------------------------------------------
# 🧹 SECTION 1: Keep only selected columns
# ---------------------------------------------

keep_columns = {col for col in df.columns if keep_column(col)}
keep_columns.update([col for col in always_keep if col in df.columns])
dropped_columns = [col for col in df.columns if col not in keep_columns]

df = df[sorted(keep_columns)]
print(f"\n🧹 Removed {len(dropped_columns)} columns not matching rules:")
for col in dropped_columns:
    print(f"  - {col}")
print(f"🔢 Shape after applying column filtering: {df.shape}")

# ---------------------------------------------
# 🚫 SECTION 2: Remove columns with missing values
# ---------------------------------------------

null_cols = df.columns[df.isnull().any()].tolist()
df = df.drop(columns=null_cols)

print(f"\n🧹 Removed {len(null_cols)} columns with missing values:")
for col in null_cols:
    print(f"  - {col}")
print(f"🔢 Shape after removing nulls: {df.shape}")

# ---------------------------------------------
# ✅ SECTION 3: Print remaining column names
# ---------------------------------------------

print("\n✅ Remaining column names:")
for col in df.columns:
    print(f"  - {col}")

# ---------------------------------------------
# 📦 SECTION 4: Separate subject info vs. model features
# ---------------------------------------------

subject_info_cols = [col for col in always_keep if col in df.columns]
df_subject_info = df[subject_info_cols].copy()
df_features = df[["subject_id"] + [col for col in df.columns if col not in subject_info_cols and col != "subject_id"]].copy()


print(f"\n📁 Separated {len(subject_info_cols)} subject info columns:")
for col in subject_info_cols:
    print(f"  - {col}")

print(f"\n📈 Remaining {df_features.shape[1]} feature columns for PCA/ML")
print(f"🔢 df_features shape: {df_features.shape}")
print(f"🧑 df_subject_info shape: {df_subject_info.shape}")

# ---------------------------------------------
# ♊ SECTION 5: Construct twin-difference file
# ---------------------------------------------
df_full = df_subject_info.merge(df_features, on='subject_id')
df_features_indexed = df_features.set_index('subject_id')
twin_diff_rows = []

for pair_id, pair_df in df_full.groupby('pair_id'):
    if len(pair_df) != 2:
        continue  # Skip incomplete twin pairs

    pair_df = pair_df.sort_values(by='twin_type_ab')
    subj_a, subj_b = pair_df.iloc[0], pair_df.iloc[1]
    id_a, id_b = subj_a['subject_id'], subj_b['subject_id']
    group_a = subj_a['clinical_group (Pr0/HP1/HC2)']
    group_b = subj_b['clinical_group (Pr0/HP1/HC2)']

    fa = df_features_indexed.loc[id_a]
    fb = df_features_indexed.loc[id_b]

    # Drop subject_id to avoid it being part of features
    fa = fa.drop(labels='subject_id', errors='ignore')
    fb = fb.drop(labels='subject_id', errors='ignore')

    if {group_a, group_b} == {0, 1}:
        if group_a == 0:
            twin_diff_rows.append({'pair_id': pair_id, 'label': 'Pr0_minus_HP1', **(fa - fb).to_dict()})
            twin_diff_rows.append({'pair_id': pair_id, 'label': 'HP1_minus_Pr0', **(fb - fa).to_dict()})
        else:
            twin_diff_rows.append({'pair_id': pair_id, 'label': 'Pr0_minus_HP1', **(fb - fa).to_dict()})
            twin_diff_rows.append({'pair_id': pair_id, 'label': 'HP1_minus_Pr0', **(fa - fb).to_dict()})

    elif group_a == 2 and group_b == 2:
        twin_diff_rows.append({'pair_id': pair_id, 'label': 'HCa_minus_HCb', **(fa - fb).to_dict()})
        twin_diff_rows.append({'pair_id': pair_id, 'label': 'HCb_minus_HCa', **(fb - fa).to_dict()})


df_twin_diffs = pd.DataFrame(twin_diff_rows)
print(f"\n✅ Twin-difference dataset shape: {df_twin_diffs.shape}")
print("🧷 Labels present:")
print(df_twin_diffs['label'].value_counts())


# Updated mapping with preferred label format
group_label_map = {
    0: "Original_Pr0",
    1: "Original_HP1",
    2: "Original_HC2"
}

# Copy original twin diffs
df_twin_full_and_diffs = df_twin_diffs.copy()

# Merge subject info and features into one place
df_full_combined = df_full.copy()

# Collect rows for original subjects
individual_rows = []

for _, row in df_full_combined.iterrows():
    subj_id = row['subject_id']
    pair_id = row['pair_id']
    clinical_group = row['clinical_group (Pr0/HP1/HC2)']

    # Create label with new format
    label = group_label_map.get(clinical_group, f"Original_{clinical_group}")

    # Drop metadata from feature row
    feature_row = row.drop(labels=[
        'participant_id', 'age', 'sex', 'group', 
        'twin_type_ab', 'clinical_group (Pr0/HP1/HC2)', 'subject_id'
    ], errors='ignore')

    row_dict = {
        'subject_id': subj_id,
        'pair_id': pair_id,
        'label': label,
        **feature_row.to_dict()
    }

    individual_rows.append(row_dict)

# Convert to DataFrame
df_individuals = pd.DataFrame(individual_rows)

# Reorder columns: subject_id first
cols = ['subject_id', 'pair_id', 'label'] + [col for col in df_individuals.columns if col not in {'subject_id', 'pair_id', 'label'}]
df_individuals = df_individuals[cols]

# Insert subject_id column (NaN) into twin-diff rows for consistency
df_twin_diffs['subject_id'] = pd.NA
df_twin_diffs = df_twin_diffs[['subject_id', 'pair_id', 'label'] + [col for col in df_twin_diffs.columns if col not in {'subject_id', 'pair_id', 'label'}]]

# Combine both
df_twin_full_and_diffs = pd.concat([df_twin_diffs, df_individuals], ignore_index=True)

# Final check
print(f"✅ Final combined dataset: {df_twin_full_and_diffs.shape}")
print("🧷 Labels present:")
print(df_twin_full_and_diffs['label'].value_counts())


#OPTIONAL but this is probably the best way to go on comparisons
#Only keep comparisons for:

#Pr0_minus_HP1 VS HP1_minus_Pr0 
#Original_HC2 vs Original_HP1
#Original_HC2 vs Original_Pr0

# Define the labels to keep
target_labels = [
    "Pr0_minus_HP1",
    "HP1_minus_Pr0",
    "Original_HC2",  
    "Original_HP1",  
    "Original_Pr0",
    "HCa_minus_HCb",
    "HCb_minus_HCa"   
]

# Filter in-place by overwriting df_twin_full_and_diffs
df_twin_full_and_diffs = df_twin_full_and_diffs[
    df_twin_full_and_diffs['label'].isin(target_labels)
].copy()

# Summary after filtering
print(f"\n✅ Filtered df_twin_full_and_diffs shape: {df_twin_full_and_diffs.shape}")
print("🧷 Labels present:")
print(df_twin_full_and_diffs['label'].value_counts())


#%%
# ---------------------------------------------
# 🧪 SECTION 5: Standardize features and apply PCA on twin differences
# ---------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Separate features and labels
feature_cols = [col for col in df_twin_full_and_diffs.columns if col not in ['label', 'pair_id', 'subject_id']]
X = df_twin_full_and_diffs[feature_cols]
y = df_twin_full_and_diffs['label']

# Step 1: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n📏 Twin-difference features standardized: mean ~0, std ~1")

# Step 2: Apply PCA (based on config)
if 'N_PCA_COMPONENTS' in globals() and N_PCA_COMPONENTS is not None:
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=PCA_RANDOM_STATE)
    pca_mode = f"{N_PCA_COMPONENTS} fixed components"
elif EXPLAINED_VARIANCE_THRESHOLD is not None:
    pca = PCA(n_components=EXPLAINED_VARIANCE_THRESHOLD, random_state=PCA_RANDOM_STATE)
    pca_mode = f"{EXPLAINED_VARIANCE_THRESHOLD:.0%} variance"
else:
    raise ValueError("❌ You must set either N_PCA_COMPONENTS or EXPLAINED_VARIANCE_THRESHOLD.")

X_pca = pca.fit_transform(X_scaled)

# Step 3: Wrap in DataFrame
pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

# Add back metadata
df_pca.insert(0, 'label', df_twin_full_and_diffs['label'].values)
df_pca.insert(0, 'pair_id', df_twin_full_and_diffs['pair_id'].values)

# Summary
print(f"\n✅ PCA completed using {pca_mode}.")
print(f"  - Original dimensions: {X_scaled.shape[1]}")
print(f"  - Reduced dimensions: {X_pca.shape[1]}")
print(f"  - Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Variance per component
print("\n🔍 Explained variance per PC:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.2%}")

# ---------------------------------------------
# 🧠 SECTION X: Feature Contributions to Each PC
# ---------------------------------------------

# Loadings: original features' contributions to each PC
pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
)

# Collect top contributors per PC
top_features_per_pc = {}
for pc in pca_loadings.columns:
    top_feats = pca_loadings[pc].abs().sort_values(ascending=False).head(10)
    top_features_per_pc[pc] = top_feats.index.tolist()
    print(f"\n📊 Top 10 features contributing to {pc}:")
    for feat in top_feats.index:
        print(f"  - {feat} (loading: {pca_loadings.loc[feat, pc]:.4f})")

# Convert top features per PC to a DataFrame for saving
top_features_df = pd.DataFrame.from_dict(top_features_per_pc, orient='index').transpose()

# Define the output path and filename
output_path = os.path.join(BASE_DIR, "Outputs ML", "Only Connectivity")
filename = "Diff_Top_Feature_Contributions.csv"
full_path = f"{output_path}\\{filename}"

# Save to CSV
top_features_df.to_csv(full_path, index=False)

print(f"\n✅ Top feature contributions saved to:\n{full_path}")


#######
import json

# Build a JSON-compatible structure with full info per PC
pca_detailed_info = {}

for i, pc in enumerate(pca_loadings.columns):
    explained_var = float(pca.explained_variance_ratio_[i])
    top_feats = pca_loadings[pc].abs().sort_values(ascending=False).head(10)
    
    pca_detailed_info[pc] = {
        'explained_variance': explained_var,
        'top_contributors': [
            {
                'feature': feat,
                'loading': float(pca_loadings.loc[feat, pc])
            }
            for feat in top_feats.index
        ]
    }

# Define the output file path
json_filename = "Diff_PCA_Detailed_Contributions.json"
json_full_path = os.path.join(output_path, json_filename)

# Save JSON
with open(json_full_path, 'w') as f:
    json.dump(pca_detailed_info, f, indent=4)

print(f"\n✅ PCA detailed contributions saved to:\n{json_full_path}")



# %%
# ---------------------------------------------
# 🤖 SECTION 6 (FINAL): Two-layer Cross Validation with Inner & Outer CV metrics and Logistic Forward
# ---------------------------------------------
import os
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set save path
save_path = os.path.join(BASE_DIR, "Outputs ML", "Only Connectivity")
os.makedirs(save_path, exist_ok=True)
print(f"📁 Output directory set to: {save_path}")

# Label preparation
label_mapping = {label: idx for idx, label in enumerate(df_twin_full_and_diffs['label'].unique())}
df_twin_full_and_diffs['label_encoded'] = df_twin_full_and_diffs['label'].map(label_mapping)

comparisons = [
    ("Pr0_minus_HP1", "HP1_minus_Pr0"),
    ("Original_HC2", "Original_HP1"),
    ("Original_HC2", "Original_Pr0"),
    ("HCa_minus_HCb", "HCb_minus_HCa")
]

# Model configurations
svc_C_values = [0.01, 0.1, 1]
rf_depths = [5, 10, 25]
gb_depths = [5, 10, 25]

model_configs = [
    ('SVC', SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'random_state': 42, 'C': c})
    for c in svc_C_values
] + [
    ('Random Forest', RandomForestClassifier, {
        'n_estimators': 100, 'max_depth': d, 'max_features': 'sqrt',
        'class_weight': 'balanced', 'random_state': 42
    }) for d in rf_depths
] + [
    ('Gradient Boosting', GradientBoostingClassifier, {
        'n_estimators': 100, 'max_depth': d, 'random_state': 42
    }) for d in gb_depths
]

# Nested CV setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_summary = []

# Run nested CV
for label1, label2 in comparisons:
    print(f"\n🔍 Starting comparison: {label1} vs {label2}")
    df_binary = df_pca[df_pca['label'].isin([label1, label2])].copy()
    X = df_binary.drop(columns=['pair_id', 'label'])
    y = df_binary['label'].map(label_mapping)

    for model_name, model_class, params in model_configs:
        print(f"\n⚙️ Model: {model_name} | Params: {params}")
        outer_scores = []

        for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"  🤁 Outer fold {fold_num+1}/5")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            start_fs = time.time()
            fs_selector = SequentialFeatureSelector(
                LogisticRegression(solver='liblinear', max_iter=1000),
                direction='forward',
                scoring='accuracy',
                n_features_to_select=min(10, X.shape[1]),
                cv=inner_cv,
                n_jobs=-1
            )
            fs_selector.fit(X_train, y_train)
            selected = X_train.columns[fs_selector.get_support()].tolist()
            print(f"    ✅ Selected features: {selected}")
            print(f"    ⏱ FS duration: {time.time() - start_fs:.2f}s")

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model_class(**params))
            ])

            print("    🔄 Running inner CV scores")
            inner_preds = []
            inner_trues = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train[selected], y_train):
                X_inner_train = X_train[selected].iloc[inner_train_idx]
                y_inner_train = y_train.iloc[inner_train_idx]
                X_inner_val = X_train[selected].iloc[inner_val_idx]
                y_inner_val = y_train.iloc[inner_val_idx]

                pipeline.fit(X_inner_train, y_inner_train)
                preds = pipeline.predict(X_inner_val)
                inner_preds.extend(preds)
                inner_trues.extend(y_inner_val)

            inner_preds = np.array(inner_preds)
            inner_trues = np.array(inner_trues)

            pipeline.fit(X_train[selected], y_train)
            y_pred = pipeline.predict(X_test[selected])

            outer_scores.append({
                'Accuracy_OuterCV': accuracy_score(y_test, y_pred),
                'Precision_OuterCV': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Recall_OuterCV': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'F1-score_OuterCV': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'Accuracy_InnerCV': accuracy_score(inner_trues, inner_preds),
                'Precision_InnerCV': precision_score(inner_trues, inner_preds, average='macro', zero_division=0),
                'Recall_InnerCV': recall_score(inner_trues, inner_preds, average='macro', zero_division=0),
                'F1-score_InnerCV': f1_score(inner_trues, inner_preds, average='macro', zero_division=0)
            })

        avg_scores = pd.DataFrame(outer_scores).mean().to_dict()
        avg_scores.update({
            'Comparison': f"{label1} vs {label2}",
            'Model': model_name,
            'Params': ', '.join(f"{k}={v}" for k, v in params.items())
        })
        cv_summary.append(avg_scores)

# Save results
print("\n📂 Saving nested CV summary table...")
df_cv_nested = pd.DataFrame(cv_summary).round(3)
summary_path = os.path.join(save_path, "NestedCV_comparison_summary.csv")
df_cv_nested.to_csv(summary_path, index=False)
print(f"📁 Saved to: {summary_path}")

# Plotting
ordered_metrics = [
    'Accuracy_OuterCV', 'Precision_OuterCV', 'Recall_OuterCV', 'F1-score_OuterCV',
    'Accuracy_InnerCV', 'Precision_InnerCV', 'Recall_InnerCV', 'F1-score_InnerCV'
]
sns.set(style="whitegrid")

unique_combos = df_cv_nested[['Model', 'Params']].drop_duplicates()
n_plots = len(unique_combos)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5), squeeze=False)
axes = axes.flatten()

for idx, (_, row) in enumerate(unique_combos.iterrows()):
    model, params = row['Model'], row['Params']
    subset = df_cv_nested[(df_cv_nested['Model'] == model) & (df_cv_nested['Params'] == params)]

    records = []
    for _, r in subset.iterrows():
        for metric in ordered_metrics:
            records.append({
                'Comparison': r['Comparison'],
                'Metric': metric,
                'Score': r[metric]
            })

    plot_data = pd.DataFrame(records)
    ax = axes[idx]
    outer = plot_data[plot_data['Metric'].str.contains('OuterCV')]
    inner = plot_data[plot_data['Metric'].str.contains('InnerCV')]
    sns.barplot(data=outer, x='Comparison', y='Score', hue='Metric', ax=ax)
    sns.barplot(data=inner, x='Comparison', y='Score', hue='Metric', ax=ax, alpha=0.5, hatch='//')

    ax.set_title(f"Model: {model}\nParams: {params}", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("Comparison")
    ax.tick_params(axis='x', rotation=30)

    plot_filename = f"NestedCV_{model.replace(' ', '_')}_{params.replace(' ', '_').replace('=', '-')}.png"
    full_plot_path = os.path.join(save_path, plot_filename)
    fig.savefig(full_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved plot: {full_plot_path}")

for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle("Nested CV Performance by Model", fontsize=14, y=1.02)
final_grid_path = os.path.join(save_path, "NestedCV_model_performance_grid.png")
plt.savefig(final_grid_path, dpi=300, bbox_inches='tight')
print(f"📈 Saved final grid plot to: {final_grid_path}")
plt.show()

df_cv_nested.head()
#%%
#Plotting again, but only with stored file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# Load from saved CSV
csv_path = os.path.join(BASE_DIR, r"Outputs ML\Only Connectivity\NestedCV_comparison_summary.csv")
df_loaded = pd.read_csv(csv_path)

# Plot settings
ordered_metrics = [
    'Accuracy_OuterCV', 'Precision_OuterCV', 'Recall_OuterCV', 'F1-score_OuterCV',
    'Accuracy_InnerCV', 'Precision_InnerCV', 'Recall_InnerCV', 'F1-score_InnerCV'
]
sns.set(style="whitegrid")

unique_combos = df_loaded[['Model', 'Params']].drop_duplicates()
n_plots = len(unique_combos)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5), squeeze=False)
axes = axes.flatten()

for idx, (_, row) in enumerate(unique_combos.iterrows()):
    model, params = row['Model'], row['Params']
    subset = df_loaded[(df_loaded['Model'] == model) & (df_loaded['Params'] == params)]

    records = []
    for _, r in subset.iterrows():
        for metric in ordered_metrics:
            records.append({
                'Comparison': r['Comparison'],
                'Metric': metric,
                'Score': r[metric]
            })

    plot_data = pd.DataFrame(records)
    ax = axes[idx]
    outer = plot_data[plot_data['Metric'].str.contains('OuterCV')]
    inner = plot_data[plot_data['Metric'].str.contains('InnerCV')]

    sns.barplot(data=outer, x='Comparison', y='Score', hue='Metric', ax=ax)
    sns.barplot(data=inner, x='Comparison', y='Score', hue='Metric', ax=ax, alpha=0.4, hatch='//')

    ax.set_title(f"Model: {model}\nParams: {params}", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("Comparison")
    ax.tick_params(axis='x', rotation=30)

# Cleanup unused axes
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

# Final formatting
plt.tight_layout()
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle("Recreated Plot: Nested CV Results", fontsize=14, y=1.02)
plt.show()

#%%
# Plotting again, but only with stored file and only accuracy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# Load from saved CSV
csv_path = os.path.join(BASE_DIR, r"Outputs ML\Only Connectivity\NestedCV_comparison_summary.csv")
df_loaded = pd.read_csv(csv_path)

# Plot settings
ordered_metrics = ['Accuracy_OuterCV', 'Accuracy_InnerCV']  # Only accuracy
sns.set(style="whitegrid")

unique_combos = df_loaded[['Model', 'Params']].drop_duplicates()
n_plots = len(unique_combos)
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5), squeeze=False)
axes = axes.flatten()

for idx, (_, row) in enumerate(unique_combos.iterrows()):
    model, params = row['Model'], row['Params']
    subset = df_loaded[(df_loaded['Model'] == model) & (df_loaded['Params'] == params)]

    records = []
    for _, r in subset.iterrows():
        for metric in ordered_metrics:
            records.append({
                'Comparison': r['Comparison'],
                'Metric': metric,
                'Score': r[metric]
            })

    plot_data = pd.DataFrame(records)
    outer = plot_data[plot_data['Metric'] == 'Accuracy_OuterCV']
    inner = plot_data[plot_data['Metric'] == 'Accuracy_InnerCV']

    ax = axes[idx]
    sns.barplot(data=outer, x='Comparison', y='Score', hue='Metric', ax=ax)
    sns.barplot(data=inner, x='Comparison', y='Score', hue='Metric', ax=ax, alpha=0.4, hatch='//')

    ax.set_title(f"Model: {model}\nParams: {params}", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Comparison")
    ax.tick_params(axis='x', rotation=30)

# Cleanup unused axes
for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

# Final formatting
plt.tight_layout()
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle("Recreated Plot: Nested CV Accuracy Only", fontsize=14, y=1.02)

# Save the figure
save_path = os.path.join(BASE_DIR, r"Outputs ML\Only Connectivity\OnlyAccuracy_plot.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%%
# ---------------------------------------------
# 🤖 SECTION 6 (FINAL): Nested CV + Summary & Rich Plots
# ---------------------------------------------

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, math, os

# Set save path
save_path = os.path.join(BASE_DIR, "Outputs ML", "Only Connectivity")
os.makedirs(save_path, exist_ok=True)
print(f"📁 Output directory set to: {save_path}")

# ---------------------------------------------
# ✅ Data Preparation
# ---------------------------------------------

assert 'label' in df_twin_full_and_diffs.columns, "df_twin_full_and_diffs must include a 'label' column"
label_mapping = {label: idx for idx, label in enumerate(df_twin_full_and_diffs['label'].unique())}
df_twin_full_and_diffs['label_encoded'] = df_twin_full_and_diffs['label'].map(label_mapping)
label_names = {v: k for k, v in label_mapping.items()}
inverse_label_names = {v: k for k, v in label_names.items()}

comparisons = [
    (inverse_label_names["Pr0_minus_HP1"], inverse_label_names["HP1_minus_Pr0"]),
    (inverse_label_names["Original_HC2"], inverse_label_names["Original_HP1"]),
    (inverse_label_names["Original_HC2"], inverse_label_names["Original_Pr0"])
]
print("✅ Label encoding and comparisons ready.")

# ---------------------------------------------
# 🛠 Model Configurations
# ---------------------------------------------

svc_C_values = [0.01, 0.1, 1]
rf_depths = [5, 10, 25]
gb_depths = [5, 10, 25]

model_configs = [
    ('SVC', SVC, {'kernel': 'rbf', 'class_weight': 'balanced', 'random_state': 42, 'C': c})
    for c in svc_C_values
] + [
    ('Random Forest', RandomForestClassifier, {
        'n_estimators': 100, 'max_depth': d, 'max_features': 'sqrt',
        'class_weight': 'balanced', 'random_state': 42
    }) for d in rf_depths
] + [
    ('Gradient Boosting', GradientBoostingClassifier, {
        'n_estimators': 100, 'max_depth': d, 'random_state': 42
    }) for d in gb_depths
]
print(f"🧱 Model configurations defined: {len(model_configs)} total.")

# ---------------------------------------------
# 🔁 Evaluation with and without CV
# ---------------------------------------------

cv_summary = []
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("🔁 Outer CV folds: 5")

for g1, g2 in comparisons:
    label1, label2 = label_names[g1], label_names[g2]
    print(f"\n📂 Working on comparison: {label1} vs {label2}")
    df_binary = df_pca[df_pca['label'].isin([label1, label2])].copy()
    X = df_binary.drop(columns=['pair_id', 'label'])
    y = df_binary['label'].map(label_mapping)
    print(f"   🔢 Number of PCA features: {X.shape[1]}")
    print(f"   🧬 Total samples: {len(X)}")

    for model_name, model_class, params in model_configs:
        print(f"🔧 Training {model_name} with params: {params}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        print(f"   🧪 Train size: {len(X_train)} | Test size: {len(X_test)}")

        start_time = time.time()
        print("   🚀 Running forward feature selection (may take time)...")
        fs_selector = SequentialFeatureSelector(
            LogisticRegression(solver='liblinear', max_iter=1000),
            direction='forward', scoring='accuracy', n_features_to_select=min(10, X.shape[1]),
            cv=3, n_jobs=-1
        )
        fs_selector.fit(X_train, y_train)
        selected = X.columns[fs_selector.get_support()].tolist()
        print(f"   ✅ Selected features ({len(selected)}): {selected}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_class(**params))
        ])
        pipeline.fit(X_train[selected], y_train)
        y_pred_test = pipeline.predict(X_test[selected])

        pipeline.fit(X[selected], y)
        y_pred_full = pipeline.predict(X[selected])

        duration = time.time() - start_time
        print(f"   🕒 Total time for model + FS: {duration:.2f} seconds")

        cv_summary.append({
            'Comparison': f"{label1} vs {label2}",
            'Model': model_name,
            'Params': ', '.join(f"{k}={v}" for k, v in params.items()),
            'Accuracy_CV': accuracy_score(y_test, y_pred_test),
            'Precision_CV': precision_score(y_test, y_pred_test, average='macro'),
            'Recall_CV': recall_score(y_test, y_pred_test, average='macro'),
            'F1-score_CV': f1_score(y_test, y_pred_test, average='macro'),
            'Accuracy_NoCV': accuracy_score(y, y_pred_full),
            'Precision_NoCV': precision_score(y, y_pred_full, average='macro'),
            'Recall_NoCV': recall_score(y, y_pred_full, average='macro'),
            'F1-score_NoCV': f1_score(y, y_pred_full, average='macro')
        })
        print(f"   ✅ Metrics collected for {model_name}")

# ---------------------------------------------
# 📋 Save Summary Table
# ---------------------------------------------

cv_df = pd.DataFrame(cv_summary)
metric_cols = [col for col in cv_df.columns if any(m in col for m in ['Accuracy', 'Precision', 'Recall', 'F1-score'])]
cv_df[metric_cols] = cv_df[metric_cols].round(2)
cv_df = cv_df.sort_values(by=['Model', 'Params', 'Comparison'])

summary_path = os.path.join(save_path, "Diff_double_cv_comparison_summary.csv")
cv_df.to_csv(summary_path, index=False)
print(f"💾 Summary table saved to: {summary_path}")

print("\n📊 Summary Table Preview:")
print(cv_df.to_string(index=False))

# ---------------------------------------------
# 📊 Rich Comparison Plots
# ---------------------------------------------

ordered_metrics = ['Accuracy_NoCV', 'F1-score_NoCV', 'Accuracy_CV', 'F1-score_CV']
sns.set(style="whitegrid")

unique_combos = cv_df[['Model', 'Params']].drop_duplicates()
n_plots = len(unique_combos)
print(f"📊 Generating plots for {n_plots} model configurations...")
n_cols = 3
n_rows = math.ceil(n_plots / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5), squeeze=False)
axes = axes.flatten()

for idx, (_, row) in enumerate(unique_combos.iterrows()):
    model, params = row['Model'], row['Params']
    subset = cv_df[(cv_df['Model'] == model) & (cv_df['Params'] == params)]

    records = []
    for _, r in subset.iterrows():
        for metric in ordered_metrics:
            records.append({
                'Comparison': r['Comparison'],
                'Metric': metric,
                'Score': r[metric]
            })

    Diff_plot_data = pd.DataFrame(records)
    ax = axes[idx]
    sns.barplot(data=Diff_plot_data, x='Comparison', y='Score', hue='Metric', hue_order=ordered_metrics, ax=ax)
    ax.set_title(f"Model: {model}\nParams: {params}", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("Comparison")
    ax.tick_params(axis='x', rotation=30)

    Diff_plot_filename = f"Diff_plot_{model.replace(' ', '_')}_{params.replace(' ', '_').replace('=', '-')}.png"
    full_Diff_plot_path = os.path.join(save_path, Diff_plot_filename)
    fig.savefig(full_Diff_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved plot to: {full_Diff_plot_path}")

for j in range(idx+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle("Model Performance by Comparison", fontsize=14, y=1.02)
final_grid_path = os.path.join(save_path, "Diff_model_performance_grid.png")
plt.savefig(final_grid_path, dpi=300, bbox_inches='tight')
print(f"📈 Saved final grid plot to: {final_grid_path}")
plt.show()


# %%
