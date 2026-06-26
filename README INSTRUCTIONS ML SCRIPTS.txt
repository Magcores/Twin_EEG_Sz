------------------------------------------------------------

This documentation has been developed by Martin Antunez (maant@dtu.dk).  
Please do not hesitate to reach out if you have any questions.

------------------------------------------------------------

Script: 6_ML_Final.py
=====================


------------------------------------------------------------
Script Overview: EEG Twin Classification Pipeline
------------------------------------------------------------

This script performs a full machine learning pipeline on EEG data 
from twin participants, designed to support clinical group comparisons 
and classification tasks.

------------------------------------------------------------
Main Steps
------------------------------------------------------------

- Load and filter EEG feature data
  Applies configurable rules to retain selected channels, frequency bands, 
  and feature types (e.g., power, FOOOF, wPLI)

- Clean missing data
  Removes columns with missing values to ensure valid input for ML models

- Construct structured datasets
  - Twin-difference rows (e.g., Pr0_minus_HP1, HP1_minus_Pr0)
  - Original subject rows (e.g., Original_HP1, Original_HC2)
  - All rows are labeled and merged into one dataset

- Apply PCA
  - Dimensionality reduction using either:
    - A fixed number of components
    - An explained variance threshold (e.g., 95%)
  - Top contributing features saved to CSV and JSON

- Run nested cross-validation
  - Inner loop: Logistic regression-based forward feature selection
  - Outer loop: Model evaluation using SVC, Random Forest, and Gradient Boosting
  - Multiple configurations tested per model

- Collect classification metrics
  Accuracy, precision, recall, and F1-score are computed for both 
  inner and outer CV folds

- Generate and save plots
  Performance plots are created for each model and comparison 
  and saved to disk

------------------------------------------------------------
Group Comparisons
------------------------------------------------------------

The script evaluates classification performance on the following group contrasts:

- Pr0_minus_HP1 vs HP1_minus_Pr0
- Original_HC2 vs Original_HP1
- Original_HC2 vs Original_Pr0
- HCa_minus_HCb vs HCb_minus_HCa

------------------------------------------------------------
Models and Hyperparameters
------------------------------------------------------------

The following classifiers are tested with multiple configurations:

- Support Vector Classifier (SVC)
  - C values: 0.01, 0.1, 1
  - Kernel: RBF
  - Class weight: balanced

- Random Forest
  - Max depth: 5, 10, 25
  - n_estimators: 100
  - max_features: sqrt
  - Class weight: balanced

- Gradient Boosting
  - Max depth: 5, 10, 25
  - n_estimators: 100

All models are wrapped in a pipeline with `StandardScaler`.


------------------------------------------------------------
Outputs
------------------------------------------------------------

- Cleaned feature dataset: df_twin_full_and_diffs
- PCA-transformed features: df_pca
- Top feature contributions:
  - Diff_Top_Feature_Contributions.csv
  - Diff_PCA_Detailed_Contributions.json
- Nested CV results:
  - NestedCV_comparison_summary.csv
  - Individual and grid performance plots (.png)
- Optional: Recreate plots directly from saved CSV

------------------------------------------------------------
All results are saved to:
<BASE_DIR>\Outputs ML
------------------------------------------------------------

Note:
Other related scripts such as "6_ML_Final_connectivity" and "6_ML_Final_periodic" 
follow the exact same pipeline, but differ only in the selected feature types.  
For example:
- "6_ML_Final_connectivity" uses only connectivity (e.g., wPLI) features
- "6_ML_Final_periodic" uses only periodic (e.g., bandpower) features

Their results are stored separately in:
- \Outputs ML\connectivity
- \Outputs ML\periodic


WARNING:
--------
All code **after** the line:

    # ---------------------------------------------
    # 🤖 SECTION 6 (FINAL): Nested CV + Summary & Rich Plots
    # ---------------------------------------------

should be **ignored**, as the final machine learning evaluation block could not be validated due to lack of access to the virtual machine environment. Only the code **before** this section is valid.

Since this script is significantly longer, it will be described in different sections:



------------------------------------------------
------------------------------------------------
------------------------------------------------
6_ML_Final.py (1)
Section: Configuration

from
# -------------------------------
# Config starts here -------------------------------
# -------------------------------
to
# -------------------------------
# Config ends here -------------------------------
# -------------------------------

------------------------------------------------------------
Configuration Logic
------------------------------------------------------------

This script includes a configurable section to control which features 
are retained for machine learning and PCA analysis. 

The logic is centralized in the keep_column() function. 
Features are grouped into types based on naming conventions. 
You can include or exclude entire groups by changing return True or return False 
within the relevant condition block.

Each feature is categorized by name (e.g., power, FOOOF, connectivity, region-level, channel-level),
and the script applies inclusion rules based on those groups.

------------------------------------------------------------
Details of Feature Groups
------------------------------------------------------------

Group 1: Aperiodic features from FOOOF – ROI-aggregated
Example: aperiodic_exponent_Global, aperiodic_offset_Frontal
Default: Excluded
To include: Set to return True

Group 2: Aperiodic features – EEG channel level
Example: aperiodic_exponent_C3, aperiodic_offset_Pz
Default: Included
To exclude: Set to return False

Group 3: Bandpower features (in dB) – ROI-aggregated
Example: alpha_db_Frontal, theta_db_Global
Default: Excluded
To include: Set to return True

Group 4: Bandpower features (in dB) – EEG channel level
Example: alpha_db_Cz, beta_db_P4
Default: Included
To exclude: Set to return False

Group 5: FOOOF peak features (CF, PW, BW) – ROI-aggregated
Example: CF_alpha_Frontal, PW_beta_Global
Default: Excluded
To include: Set to return True

Group 6: FOOOF peak features (CF, PW, BW) – EEG channel level
Example: CF_alpha_C3, BW_gamma_Pz
Default: Included
To exclude: Set to return False

Group 7: wPLI connectivity features
Example: wpli_alpha_Pz_Cz, wpli_gamma_T7_T8
Default: Included
To exclude: Set to return False

------------------------------------------------------------
Always-Kept Metadata Columns
------------------------------------------------------------

These metadata columns are always retained (if present), 
regardless of keep_column() logic:

participant_id  
age  
sex  
group  
pair_id  
twin_type_ab  
clinical_group (Pr0/HP1/HC2)  
subject_id

These are essential for twin pairing, group comparisons, 
and tracking individual information during downstream steps.

------------------------------------------------------------
PCA Configuration
------------------------------------------------------------

Dimensionality reduction using PCA is controlled with two options:

Option 1: Explained variance threshold (recommended)
Variable: EXPLAINED_VARIANCE_THRESHOLD  
Example: 0.95 retains components that explain 95% of the variance

Option 2: Fixed number of components
Variable: N_PCA_COMPONENTS  
Example: 30 keeps the first 30 components

Note: Only one of these options should be set at a time. 
The other must be set to None.

------------------------------------------------------------
Default Selection Summary
------------------------------------------------------------

By default, only EEG channel-level features are included. 
ROI-aggregated features are excluded.

This is based on the assumption that channel-level data 
preserves more spatial detail, useful for twin comparisons and ML.

Summary of default inclusion:

Aperiodic (FOOOF): Channel-level = Included, ROI = Excluded  
Bandpower (dB): Channel-level = Included, ROI = Excluded  
FOOOF Peaks (CF, PW, BW): Channel-level = Included, ROI = Excluded  
wPLI Connectivity: Included

To change inclusion, edit the return values in the keep_column() function.

------------------------------------------------------------
Suggestions for Custom Feature Selection
------------------------------------------------------------

You can tailor the feature set by modifying the return logic in keep_column() 
to match specific research or modeling goals. Below are some common configurations 
you might consider:

1. Use Only Power Features
   - Include only bandpower features (set return True for Groups 3 and/or 4)
   - Exclude FOOOF and connectivity features (set return False for Groups 1, 2, 5, 6, and 7)

2. Use Only Connectivity Features
   - Keep only Group 7 (wPLI features)
   - Exclude all other groups (Groups 1–6)

3. Focus on Specific Frequency Bands
   - Add custom logic to include features only for desired bands
     (e.g., include only alpha or theta by checking for 'alpha' or 'theta' in the name)

4. Use Only ROI-Aggregated Features
   - Include Groups 1, 3, and 5
   - Exclude Groups 2, 4, and 6 (channel-level features)

5. Use Only Channel-Level Features
   - Keep Groups 2, 4, 6, and 7 (default setup)
   - Exclude Groups 1, 3, and 5

6. Combine FOOOF and Power Features
   - Include both bandpower (Groups 3/4) and FOOOF features (Groups 1/2 and/or 5/6)
   - Optionally filter by level (ROI or channel)

7. Minimalist Setup for Fast Testing
   - Include just one type of feature (e.g., only alpha power at channel level)
   - Useful for debugging or benchmarking

8. Custom Selection by Brain Region or Channel
   - Add checks to include only features from specific ROIs (e.g., 'Frontal') or channels (e.g., 'Pz', 'C3')

Feel free to combine these strategies to match your specific hypothesis 
or simplify the input for faster training and interpretation.

------------------------------------------------
------------------------------------------------
------------------------------------------------
6_ML_Final.py (2)
from
# ---------------------------------------------
# 🧹 SECTION 1: Keep only selected columns
# ---------------------------------------------
to (including)
# ---------------------------------------------
# ♊ SECTION 5: Construct twin-difference file
# ---------------------------------------------

------------------------------------------------------------
Section2: EEG Dataset Filtering and Twin Difference Construction
------------------------------------------------------------

This section prepares EEG data for downstream machine learning by:
- Filtering features based on naming logic
- Removing columns with missing values
- Creating structured datasets for both individual subjects and twin-pair differences

------------------------------------------------------------
Inputs
------------------------------------------------------------

- Raw EEG feature DataFrame  
  Includes both EEG-derived features and metadata columns.

- keep_column() function  
  Defines inclusion logic for different feature types (e.g., power, FOOOF, connectivity)

------------------------------------------------------------
Outputs
------------------------------------------------------------

- df_twin_full_and_diffs  
  Final dataset containing:
    - Twin-difference feature rows (paired comparisons)
    - Original subject feature rows (individual-level data)
    - Cleaned EEG feature columns (no missing values)
    - Labels for classification

------------------------------------------------------------
Processing Steps
------------------------------------------------------------

1. **Filter EEG Feature Columns**

   - Applies keep_column() logic to select relevant features
   - Removes features not matching desired types or structure (e.g., ROI vs channel)
   - Always retains essential metadata columns:

     participant_id  
     age  
     sex  
     group  
     pair_id  
     twin_type_ab  
     clinical_group  
     subject_id

2. **Remove Columns with Missing Values**

   - Drops any EEG feature columns that contain NaN values
   - Ensures all remaining features are valid for machine learning input

3. **Split Features and Metadata**

   - Separates the dataset into:

     - df_features: EEG features only (numeric)
     - df_subject_info: metadata (demographic, clinical, etc.)

4. **Create Twin-Difference Feature Rows**

   - For each complete twin pair (identified by pair_id):

     - Computes the difference between the twins’ EEG feature vectors
     - Assigns a classification label based on group membership:

       Pr0_minus_HP1  
       HP1_minus_Pr0  
       HCa_minus_HCb  
       HCb_minus_HCa

   - These rows represent within-pair contrasts for classification tasks

5. **Append Individual Subject Rows**

   - Re-adds each original subject as a standalone row
   - Assigns group-level labels:

     Original_Pr0  
     Original_HP1  
     Original_HC2

   - Removes metadata from feature columns for consistency

6. **Merge All Rows into a Final Dataset**

   - Combines twin-difference and individual subject rows
   - Final DataFrame includes:

     - label  
     - pair_id  
     - subject_id (NaN for difference rows)  
     - Cleaned EEG feature columns

7. **(Optional) Filter by Target Labels**

   - Keeps only specific labels for focused classification tasks:

     Pr0_minus_HP1  
     HP1_minus_Pr0  
     Original_Pr0  
     Original_HP1  
     Original_HC2  
     HCa_minus_HCb  
     HCb_minus_HCa

------------------------------------------------------------
Final Output
------------------------------------------------------------

- A unified dataset (`df_twin_full_and_diffs`) containing:

  - EEG features ready for standardization, PCA, or ML training  
  - No missing values or irrelevant columns  
  - Labels for both pairwise and individual classification tasks  
  - Flexible structure for downstream analysis

This dataset forms the foundation for all subsequent modeling steps.



------------------------------------------------
------------------------------------------------
------------------------------------------------
6_ML_Final.py (3)
# ---------------------------------------------
# 🧪 SECTION 5: Standardize features and apply PCA on twin differences
# ---------------------------------------------
-----------------------------------------------------------------------
Section : Standardization and PCA on Twin Differences
------------------------------------------------------------

This section reduces the dimensionality of the EEG twin-difference dataset 
by standardizing features and applying PCA (Principal Component Analysis). 
It also identifies the most influential original features for each principal component 
and saves these results for interpretation and reporting.

------------------------------------------------------------
Inputs
------------------------------------------------------------

- df_twin_full_and_diffs  
  A DataFrame combining individual and twin-difference EEG feature rows.  
  Expected columns include:  
    - EEG feature columns (numerical)  
    - label  
    - pair_id  
    - subject_id (optional for difference rows)

- Configuration parameters (must set one of the two):  
    - N_PCA_COMPONENTS (integer): number of principal components to keep  
    - EXPLAINED_VARIANCE_THRESHOLD (float): target variance to retain (e.g., 0.95)  
    - PCA_RANDOM_STATE: integer seed for reproducibility

------------------------------------------------------------
Outputs
------------------------------------------------------------

Files are saved to the following location:

<BASE_DIR>\Outputs ML

Generated files:

- df_pca  
  PCA-transformed feature DataFrame (used for downstream ML)  
  Includes principal components (PC1, PC2, ...) along with label and pair_id

- Diff_Top_Feature_Contributions.csv  
  Summary of the top 10 contributing features for each principal component  
  Full path:  
  <BASE_DIR>\Outputs ML\Diff_Top_Feature_Contributions.csv

- Diff_PCA_Detailed_Contributions.json  
  Detailed PCA output with explained variance and top feature loadings per PC  
  Full path:  
  <BASE_DIR>\Outputs ML\Diff_PCA_Detailed_Contributions.json

------------------------------------------------------------
Processing Steps
------------------------------------------------------------

1. **Select EEG Feature Columns**
   - Removes non-feature columns (label, pair_id, subject_id)
   - Stores remaining EEG features in `X`, and labels in `y`

2. **Standardize Features**
   - Uses `StandardScaler` to normalize feature values
   - Transforms all columns to have mean ~0 and standard deviation ~1
   - Ensures all features contribute equally to PCA

3. **Apply PCA**
   - Reduces feature dimensionality using PCA
   - Retains components based on configuration:

     a. **Fixed number of components**  
        Set `N_PCA_COMPONENTS` to a desired integer  
        Example: 30 components

     b. **Explained variance threshold**  
        Set `EXPLAINED_VARIANCE_THRESHOLD` to a float between 0 and 1  
        Example: 0.95 keeps enough components to explain 95% of total variance

   - PCA-transformed values are stored in `df_pca`  
     Columns are named: PC1, PC2, ..., PCn  
     Metadata columns (`pair_id`, `label`) are re-attached

4. **Print PCA Summary**
   - Displays PCA mode used, number of dimensions before/after
   - Prints total explained variance and variance per principal component

5. **Analyze Feature Contributions (Loadings)**
   - Calculates how much each original feature contributes to each principal component
   - Extracts top 10 contributing features for each PC
   - Prints the top contributors for review

6. **Save Output Files**
   - Writes two summary files to the shared project folder:

     a. `Diff_Top_Feature_Contributions.csv`  
        - Tabular summary of top 10 features per PC  
        - One column per PC with top contributors listed

     b. `Diff_PCA_Detailed_Contributions.json`  
        - JSON structure with:
          - Explained variance per PC
          - Top 10 feature loadings (values and names) per PC

------------------------------------------------------------
Remarks
------------------------------------------------------------

- PCA helps reduce overfitting and improves model efficiency 
  by transforming EEG features into a smaller, decorrelated set.

- The saved contribution files support interpretation of 
  which original EEG measures are most influential in the data.

- Be sure to set either `N_PCA_COMPONENTS` or `EXPLAINED_VARIANCE_THRESHOLD`. 
  Leaving both unset will cause an error.

- PCA configuration should reflect your balance between 
  interpretability and model performance.

------------------------------------------------
------------------------------------------------
------------------------------------------------
6_ML_Final.py (4)
--------------------------------------------------
# ---------------------------------------------
# 🤖 SECTION 6 (FINAL): Two-layer Cross Validation with Inner & Outer CV metrics and Logistic Forward
# ---------------------------------------------
------------------------------------------------------------
Section: Nested Cross-Validation with Feature Selection and Model Evaluation
------------------------------------------------------------

This section runs a two-layer cross-validation framework to evaluate different models 
on selected classification tasks. It includes inner-loop feature selection using 
logistic regression and outer-loop evaluation across multiple algorithms and hyperparameters.

------------------------------------------------------------
Inputs
------------------------------------------------------------

- df_pca  
  DataFrame with PCA-transformed EEG features + labels + pair_id

- Comparisons (class label pairs):  
  - Pr0_minus_HP1 vs HP1_minus_Pr0  
  - Original_HC2 vs Original_HP1  
  - Original_HC2 vs Original_Pr0  
  - HCa_minus_HCb vs HCb_minus_HCa

- Models tested:
  - Support Vector Classifier (SVC) with different `C` values
  - Random Forest with different tree depths
  - Gradient Boosting with different tree depths

- Configuration:
  - Outer CV: 5-fold StratifiedKFold (model evaluation)
  - Inner CV: 3-fold StratifiedKFold (feature selection)
  - LogisticRegression used for sequential forward feature selection
  - Top 10 features selected (or fewer if limited by input dimensionality)

------------------------------------------------------------
Outputs
------------------------------------------------------------

Saved to:
<BASE_DIR>\Outputs ML

- NestedCV_comparison_summary.csv  
  Table with average inner and outer CV scores for each model + comparison

- NestedCV_model_performance_grid.png  
  Grid of bar plots showing model performance across all comparisons

- NestedCV_<Model>_<Params>.png (one per model)  
  Individual performance plots for each model/parameter combination

------------------------------------------------------------
Processing Steps
------------------------------------------------------------

1. **Label Encoding and Dataset Preparation**
   - Encodes labels numerically for classification
   - Filters the dataset for each binary comparison

2. **Outer Loop (Model Evaluation)**
   - Splits data into training/test using 5-fold StratifiedKFold
   - For each fold:
     - Runs inner loop to select features
     - Evaluates model on the held-out test fold using selected features

3. **Inner Loop (Feature Selection)**
   - Applies forward selection with Logistic Regression
   - Uses 3-fold StratifiedKFold on training data to select up to 10 best features

4. **Model Training and Prediction**
   - Trains pipeline (StandardScaler + model) on selected features
   - Computes inner and outer CV predictions
   - Metrics collected:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - All metrics are computed separately for inner and outer folds

5. **Score Aggregation**
   - Averages metrics across outer folds
   - Stores results along with model name and parameter settings

6. **Save Summary Table**
   - Outputs all results to CSV for recordkeeping and later visualization

7. **Plotting Results**
   - Generates individual bar plots comparing inner vs outer CV scores
   - One plot per model/parameter combination
   - Saves as PNG files

8. **Final Summary Grid**
   - Combines all individual plots into a multi-panel figure
   - Saves overall summary as NestedCV_model_performance_grid.png

------------------------------------------------------------
(Extra) Re-Plotting from Saved CSV
------------------------------------------------------------

- Reloads the summary CSV
- Recreates the same plots without rerunning CV
- Useful for sharing or updating visuals without long runtime

------------------------------------------------------------
Remarks
------------------------------------------------------------

- This nested CV approach avoids overfitting by separating feature selection 
  from model evaluation.

- Inner CV uses Logistic Regression for stable feature ranking, 
  regardless of the final classifier.

- Outer CV scores reflect real-world generalization performance.

- The summary CSV and plots help compare models across different clinical contrasts 
  (e.g., discordant twins, healthy controls, etc.).

- Highly flexible and scalable for trying new models or configurations.


-------------------------
-------------------------
-------------------------

BONUS
#Plotting again, but only with stored file
-------------------------
------------------------------------------------------------
Section: Re-Plotting Nested CV Results from Stored CSV
------------------------------------------------------------

This section reloads previously saved nested cross-validation results 
and regenerates the model performance plots without re-running the full pipeline.

Useful for quickly visualizing model performance, sharing figures, or 
updating plots after modifications to layout or styling.

------------------------------------------------------------
Input
------------------------------------------------------------

- CSV file: NestedCV_comparison_summary.csv  
  Full path:  
  <BASE_DIR>\Outputs ML\NestedCV_comparison_summary.csv

  This file must contain:
    - Comparison (e.g., Pr0_minus_HP1 vs HP1_minus_Pr0)  
    - Model and hyperparameters  
    - CV metrics: accuracy, precision, recall, F1 for both inner and outer folds

------------------------------------------------------------
Output
------------------------------------------------------------

- Re-created performance plots (shown in a multi-panel figure)  
  Each subplot displays inner and outer CV scores across comparisons  
  for a specific model + parameter configuration

- No files are saved in this step (visualization only)

------------------------------------------------------------
Steps Performed
------------------------------------------------------------

1. **Load CSV File**
   - Reads in the previously saved summary CSV with performance metrics

2. **Plot Setup**
   - Defines metrics to be plotted:
     - Accuracy, Precision, Recall, F1 (inner and outer CV)
   - Calculates number of plots and grid layout

3. **Iterate Over Model-Parameter Combinations**
   - For each unique combination:
     - Filters data
     - Converts into long format for bar plotting
     - Separates inner vs outer CV results
     - Plots side-by-side bars for each metric, per comparison
     - Adds model name and parameters as subplot titles

4. **Final Plot Formatting**
   - Removes unused axes if model count is not divisible by number of columns
   - Applies consistent axis formatting and rotations
   - Adds legend and overall title

5. **Show Plot**
   - Displays the final plot grid with all model comparisons
   - Allows user to visually compare performance across models and metrics

------------------------------------------------------------
Remarks
------------------------------------------------------------

- This re-plotting step avoids the need to re-run time-consuming nested CV.
- Ensures reproducible visualization of previously saved results.
- To export images, use `plt.savefig()` manually if needed.
- Ideal for exploratory review or presentation of results.
