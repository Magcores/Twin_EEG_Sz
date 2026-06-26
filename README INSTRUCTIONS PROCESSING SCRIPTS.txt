------------------------------------------------------------

This documentation has been developed by Martin Antunez (maant@dtu.dk).  
Please do not hesitate to reach out if you have any questions.


------------------------------------------------------------
Introduction
------------------------------------------------------------

This document describes the full EEG analysis pipeline used in the Twin Study (SZ project).
It covers every step from raw data conversion to final machine learning evaluation,
including preprocessing, source localization, feature extraction, and dataset preparation.

Each section explains the logic, input/output paths, dependencies, and purpose of a specific script.

All scripts are designed to be modular and reproducible, with intermediate and final outputs saved
in clearly defined directories for inspection and reuse.

------------------------------------------------------------
Scripts Overview (in execution order)
------------------------------------------------------------

0_BIDS.py  
    - Converts .fif files to .edf and organizes data in BIDS format.

1_PreprocessingRT2.0.py  
    - Preprocesses raw EEG (filtering, ICA, artifact rejection, QC reports).

2_SourceLocalization.py  
    - Applies source modeling (dSPM), producing cortical ROI signals.

3_wPLI.py  
    - Computes wPLI-based functional connectivity across EEG frequency bands.

4_Power&FOOOF.py (missing, methodology fully described)  
    - Extracted spectral features (FOOOF) from ROI/channel time series.
    - Outputs already exist and are used in later scripts.

5_1_PuttingFeaturesTogether_Power.py  
    - Aggregates spectral features per subject, per ROI, and globally.

5_2_PuttingFeaturesTogether_Connectivity.py  
    - Aggregates wPLI connectivity matrices into long/wide format.

5_3_PuttingFeaturesTogether_wplifooof.py  
    - Merges spectral and connectivity features into one dataset.

5_4_Adding subject data.py  
    - Appends subject metadata (from participants.tsv) to the final feature matrix.

6_ML_Final.py  
    - This is described in "README INSTRUCTIONS ML SCRIPTS.txt"


------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------



Script: 0_BIDS.py
=================

Convert .fif to .edf and BIDS Format
------------------------------------

This script processes EEG recordings by:

  1. Converting .fif files to .edf format.
  2. Reorganizing .edf files into BIDS-compliant structure using MNE-BIDS.
  3. Creating a participants.tsv metadata file from an Excel spreadsheet.


Input and Output Paths
----------------------

Input: Raw .fif files
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Raws

Output: Converted .edf files
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Raw_edf

Output: BIDS-compliant structure
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\BIDS

Input: Metadata Excel file
  Path: P:\<HOSPITAL_NETWORK>\
        <researcher>\VIP_rsEEG\VIP_MASTER_selected_variables_270325.xls

Output: BIDS metadata file (participants.tsv)
  Path: [Same as BIDS root]/participants.tsv


Dependencies
------------

Install these Python packages from the terminal before running the script:

  pip install mne mne-bids
  pip install edfio        (optional, for EDF writing)
  pip install pandas xlrd  (for Excel reading)


Script Functionality
--------------------

Loop 1: Convert .fif to .edf
  - Scans the input folder for .fif files.
  - Converts each file to .edf unless it already exists.
  - Saves the .edf files into the designated output folder.

Loop 2: Convert .edf to BIDS format
  - Loads each .edf file and writes it into a BIDS-compliant structure.
  - Output structure example:
      sub-XXXXXXX/ses-01/eeg/sub-XXXXXXX_ses-01_task-rest_eeg.edf

Step 3: Create participants.tsv
  - Extracts metadata from Excel file using subject IDs.
  - Filters to include only participants with corresponding .edf files.
  - Creates a BIDS-compliant participants.tsv file with the following columns:

      participant_id
      sex
      age
      group
      pair_id
      twin_type_ab
      clinical_group (Pr0/HP1/HC2)


Important Notes
---------------

- Subject IDs are derived from the first 7 characters of the "ID" field.
- IDs are padded to ensure they are 7 characters long (e.g., '0012345').
- Session and task values are fixed: ses-01, task-rest.
- BIDS output is skipped if the file already exists.
- The participants.tsv file will only include subjects that have .edf data.


Output Summary
--------------

After running the full script, you should have:

  - .edf files in the Raw_edf directory
  - A BIDS-compliant folder structure in the BIDS directory
  - A participants.tsv file located at the root of the BIDS folder



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------



Script: 1_PreprocessingRT2.0.py
================================

Resting-State EEG Preprocessing Pipeline (Twins SZ Project)
------------------------------------------------------------

This script performs complete preprocessing of raw EEG data, including:

  1. Processing one or multiple subjects from the raw EEG directory
  2. Resampling, filtering, bad channel detection, ICA, and artifact rejection
  3. Saving preprocessed continuous and epoched data
  4. Generating quality control (QC) plots and tables
  5. Writing subject-specific metadata and a master QC summary

Author: Martin Antunez


Input and Output Paths
-----------------------

Input: Raw EEG files (.fif format)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Raws

Output: Preprocessed continuous EEG
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Continuous\subID_preprocessed_raw.fif

Output: Epoched EEG data
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Epoched\subID_epoched-epo.fif

Output: QC plots and metadata
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Plots\subID\

Output: Master QC summary table
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Plots\qc_summary_all_subjects.csv


Dependencies
------------

Install the following Python packages:

  pip install mne mne_icalabel autoreject pandas matplotlib scipy numpy


Script Functionality
--------------------

- Automatically detects all *_raw.fif files in the Raws directory
- Skips subjects that are already processed
- For each subject:

  * Loads raw EEG
  * Resamples to 256 Hz and applies 1–100 Hz bandpass + notch filter at 50 Hz
  * Detects bad channels using z-scored PSD (1–45 Hz)
  * Saves PSD plot with bad channels highlighted
  * Interpolates bad channels and applies average reference
  * Runs ICA with ICLabel to remove non-brain components
  * Applies AutoReject to clean epochs (2-second segments)
  * Saves cleaned raw and epoched data
  * Generates and saves:
      - Overview plots
      - ICA component plots (if components are removed)
      - AutoReject logs
      - Table of consistently bad channels
      - JSON metadata summary

- Adds or updates a master QC CSV with all subjects' metadata and QC results


Important Notes
---------------

- Set `process_all_subjects = True` to process all available subjects
- Individual subjects can be tested by setting the flag to False
- Bad channel detection is based on a PSD z-score threshold (> 2.5)
- ICA uses the ICLabel method to retain only components labeled as 'brain'
- Consistently bad channels are defined as those interpolated in >90% of epochs
- All QC files and metadata are stored under each subject’s plot directory


QC Outputs Per Subject
----------------------

Each subject will generate:

  - Raw EEG overview plot
  - PSD plot with bad channels highlighted
  - ICA component plot (if non-brain components excluded)
  - AutoReject log plot
  - CSV file listing consistently bad channels
  - JSON file with summary metadata


Final Output Summary
---------------------

After running the script, you will have:

  - Cleaned continuous EEG files (.fif) in "Preprocessed\Continuous"
  - Cleaned epoched EEG files (.fif) in "Preprocessed\Epoched"
  - Plots and QC data in "Preprocessed\Plots\subID"
  - A group-wide QC summary in "Preprocessed\Plots\qc_summary_all_subjects.csv"



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------


Script: 2_SourceLocalization.py
===============================

Source Localization for Resting-State EEG (Twins SZ Project)
------------------------------------------------------------

This script performs cortical source localization on preprocessed EEG data using MNE-Python.
It applies the dSPM inverse method with the fsaverage template brain and outputs:

  1. Source estimates (STC files)
  2. ROI-averaged time series (based on aparc parcellation)
  3. Mean activity summary for each subject
  4. Group-wide processing summary table

Optional 3D visualization tools (PyVista, ipywidgets) are supported but not required.


Input and Output Paths
-----------------------

Input: Preprocessed EEG files (after artifact cleaning)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Continuous\subID_preprocessed_raw.fif

Output: Source estimates (STC)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Source\subID\subID_source_dSPM-lh.stc

Output: ROI time courses (per subject)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Source\subID\subID_roi_timecourses.csv

Output: Subject summary (mean activity over time)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Source\subID\source_summary.csv

Output: Group-level summary (accumulated)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Source\source_summary_all_subjects.csv

Template brain (fsaverage)
  Path: P:\...\Preprocessed\Source\fs_templates\fsaverage


Dependencies
------------

Required:

  pip install mne pandas matplotlib

Optional (for 3D visualizations):

  pip install pyvistaqt qtpy
  pip install ipywidgets ipyevents

Or via conda:

  conda install -c conda-forge pyvistaqt qtpy ipywidgets ipyevents


Script Functionality
--------------------

- Loads each subject’s preprocessed EEG data
- Applies average reference (required for inverse modeling)
- Optionally downsamples to 100 Hz to reduce memory usage
- Loads fsaverage template (head model, surfaces, BEM)
- Computes forward model and noise covariance
- Applies inverse method (dSPM) to compute source time courses
- Saves left and right hemisphere STC files
- Extracts mean cortical activity per time point → saves as CSV
- Extracts ROI time courses based on aparc parcellation
- Saves ROI signals (time x region matrix) as CSV


Important Notes
---------------

- Uses `fsaverage` template for head model and source space
- Source space: oct6 (approx. 4.9mm spacing)
- BEM: 3-layer (brain, CSF, skull) conductivity model
- Inverse method: dSPM (can be changed via `inverse_method`)
- SNR used: 3.0 (lambda² = 1 / SNR²)
- Automatically skips subjects already processed unless `force_rerun = True`
- ROI extraction removes labels with no overlap in source space or labeled "unknown"


Output Files Per Subject
------------------------

Each subject will have the following in their source folder:

  - STC source estimate: subID_source_dSPM-lh.stc / -rh.stc
  - ROI time courses CSV: subID_roi_timecourses.csv
  - Mean activity summary CSV: source_summary.csv


Final Output Summary
---------------------

After running the script, you will have:

  - Source localization results for each subject in "Preprocessed\Source"
  - ROI-based time courses (parcellated activity)
  - Subject summaries and one master CSV of all subjects
  - (Optional) 3D brain visualization if optional dependencies are installed



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------


Script: 3_wPLI.py
=================

Resting-State Functional Connectivity (wPLI)
--------------------------------------------

This script computes functional connectivity using the **weighted Phase Lag Index (wPLI)**
from ROI-based source time courses (output of source localization). It performs:

  1. Band-pass filtering into standard EEG frequency bands
  2. Pairwise wPLI calculation between all ROI signals
  3. Saving subject-level connectivity matrices and heatmaps
  4. Group-averaging of connectivity matrices per band


Input and Output Paths
-----------------------

Input: ROI time courses per subject (CSV format)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Source\subID\subID_roi_timecourses.csv

Output: Subject-level wPLI matrices and plots
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Connectivity_wPLI_bands\subID\
        wpli_<band>.csv
        wpli_<band>_heatmap.png

Output: Group-average connectivity
  Path: P:\...\Connectivity_wPLI_bands\group_mean_wpli_<band>.csv
        P:\...\Connectivity_wPLI_bands\group_mean_wpli_<band>_heatmap.png


Dependencies
------------

Required:

  pip install numpy pandas scipy matplotlib seaborn tqdm


Script Functionality
--------------------

For each subject:
  - Loads the ROI time series (from source localization)
  - Applies 4th-order Butterworth band-pass filter for each frequency band
  - Computes pairwise wPLI using the Hilbert transform for each band
  - Outputs:
      - wPLI matrix as CSV
      - Heatmap plot for visual inspection

At the end:
  - Loads all individual wPLI matrices
  - Computes group-average wPLI matrix per band
  - Saves group mean matrix and heatmap


Frequency Bands Used
--------------------

  - Delta: 1–4 Hz
  - Theta: 4–8 Hz
  - Alpha: 8–13 Hz
  - Beta: 13–30 Hz
  - Gamma: 30–45 Hz


Important Notes
---------------

- Sampling frequency is assumed to be 100 Hz
- Input ROI files must have:
    - First column = time
    - Remaining columns = ROI signals
- All wPLI matrices are symmetric (i.e., [i,j] = [j,i])
- Diagonal values are set to 0
- Subjects already processed are skipped unless their output folders are removed
- All computations are wrapped in error handling to skip failed subjects


Output Files Per Subject
------------------------

Each subject's folder in `Connectivity_wPLI_bands` contains:

  - wpli_delta.csv
  - wpli_theta.csv
  - wpli_alpha.csv
  - wpli_beta.csv
  - wpli_gamma.csv

  - wpli_<band>_heatmap.png (one for each band)


Final Output Summary
---------------------

After running the script, you will have:

  - Subject-level connectivity matrices and plots for each band
  - Group-average matrices and plots in the root of the output folder
  - Ready-to-use data for graph-theory analysis, network statistics, etc.



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------

Script: 4_Power&FOOOF.py (LOST, rewritting will be needed, instructions are provided)
===============================

Power Spectrum Estimation and FOOOF Feature Extraction
-------------------------------------------------------

This script is no longer available (Unfortunately, it seems to 
have been rewritten by the content of 3_wPLI.py), 
however,  the **full methodology was strictly implemented**
as described in **Section 2.2 of the thesis**. The process involved detailed extraction of
spectral features from each ROI/channel using a combination of **Welch’s method** and the
**FOOOF (Fitting Oscillations & One-Over-F)** model.

Although the exact implementation is not preserved, the following outlines the methodology
and expected output format, allowing the script to be reconstructed in the future if needed.

Luckily, outputs from the lost scripts were already available at:
P:\...\Preprocessed\PowerAndFoooF\subID_features.csv

Therefore, later scripts (5 and 6) can be run.


Methodology (as implemented)
----------------------------

1. **Power Spectrum Estimation** (Welch’s Method)
   - Signals were segmented into overlapping windows
   - Hamming window applied to each segment
   - Fast Fourier Transform (FFT) performed
   - Periodograms averaged to estimate the Power Spectral Density (PSD)
   - Frequency resolution and overlap as defined in Section 3.5

2. **FOOOF Fitting**
   - Each PSD was modeled as a sum of:
       - Aperiodic (1/f-like) background
       - Gaussian-shaped periodic peaks
   - Fitting performed in semi-log space (linear frequencies, log power)
   - FOOOF parameters extracted:
       - **Aperiodic**: slope (exponent), offset (intercept)
       - **Periodic**: frequency peaks modeled with:
           - Center Frequency (CF)
           - Power (PW)
           - Bandwidth (BW)
   - Additional metrics: goodness of fit (R²) and error

3. **Total Band Power Extraction**
   - For standard EEG bands (delta, theta, alpha, beta, gamma):
       - Total power computed in **linear and dB scale**
       - Extracted from raw PSD, not FOOOF decomposition

4. **Peak Selection**
   - For each frequency band, the **strongest detected peak** (highest PW) within that range
     was retained, if available.
   - If no peak was detected in a given band, values were marked as missing (NaN).


Expected Output Files
---------------------

Location:
  P:\...\Preprocessed\PowerAndFoooF\subID_features.csv

Each output CSV contains **one row per detected spectral peak per channel**.
Expected columns include:

  - subject_id
  - channel

  - aperiodic_offset
  - aperiodic_exponent
  - r_squared
  - fit_error

  - delta_lin, delta_dB
  - theta_lin, theta_dB
  - alpha_lin, alpha_dB
  - beta_lin, beta_dB
  - gamma_lin, gamma_dB

  - CF_delta, PW_delta, BW_delta
  - CF_theta, PW_theta, BW_theta
  - CF_alpha, PW_alpha, BW_alpha
  - CF_beta,  PW_beta,  BW_beta
  - CF_gamma, PW_gamma, BW_gamma


Dependencies
----------------------

  - numpy
  - pandas
  - scipy
  - matplotlib
  - fooof


Relevance to Pipeline
---------------------

This script directly produced the inputs required by:

  5_1_PuttingFeaturesTogether_Power.py

That script expects precomputed features per subject and per channel in the format above,
and will fail if those files are missing or malformed.


Recommendations for Rebuilding
------------------------------

To recreate this script:

  1. **Load ROI/channel time series**
     - Source data should be in CSV format (1 column per ROI)
     - Sampled at 100 Hz

  2. **Compute Power Spectral Density (PSD)**
     - Use `scipy.signal.welch()` with:
         - Hamming window
         - Overlapping segments (e.g., 50% overlap)
         - FFT length and window length as per Section 3.5
     - Output frequency range: 1–45 Hz

  3. **Fit FOOOF to each PSD**
     - Use `fooof.FOOOF()` to fit the power spectrum in semi-log space
     - Extract:
         - Aperiodic parameters: `aperiodic_offset`, `aperiodic_exponent`
         - Periodic peaks: `CF`, `PW`, `BW` for each detected peak
         - Model fit metrics: `r_squared`, `fit_error`

  4. **Extract total power in standard EEG bands**
     - Use band ranges:
         - delta: 1–4 Hz
         - theta: 4–8 Hz
         - alpha: 8–13 Hz
         - beta: 13–30 Hz
         - gamma: 30–45 Hz
     - For each band:
         - Integrate PSD over the band to compute **total power**
         - Save both:
             - `band_lin` = linear power
             - `band_dB`  = 10 × log10(linear power)

  5. **Extract strongest peak per band**
     - For each band, identify all FOOOF peaks where `CF` falls within the band
     - Retain the peak with the highest `PW` (power)
     - If no peak exists in a band, set `CF`, `PW`, and `BW` to NaN for that band

  6. **Save results per subject**
     - Output file: `subID_features.csv`
     - One row per channel, including all fields:
         - subject_id
         - channel
         - aperiodic_offset, aperiodic_exponent, r_squared, fit_error
         - Total power per band: `delta_lin`, `delta_dB`, etc.
         - Per-band peak parameters:
             - CF_delta, PW_delta, BW_delta
             - CF_theta, PW_theta, BW_theta
             - ... (same for alpha, beta, gamma)

This output is required for `5_1_PuttingFeaturesTogether_Power.py` to run correctly.
(Previous outputs from previous runs are already provided in P:\...\Preprocessed\PowerAndFoooF\subID_features.csv)

----------

This documentation ensures full reproducibility even without access to the original script.




------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------


Script: 5_1_PuttingFeaturesTogether_Power.py
============================================

Aggregating Spectral Features Across Channels, ROIs, and Subjects
------------------------------------------------------------------

This script processes per-channel spectral features (from FOOOF analysis) and produces:

  1. Peak-based power features per frequency band (per channel)
  2. ROI-averaged and global-averaged summaries
  3. Final subject-level wide-format table for statistics or ML


Input and Output Paths
-----------------------

Input: Per-channel spectral features (per subject)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\PowerAndFoooF\subID_features.csv

Output: Aggregated tables
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Features\

    - fooof_AllChannels.csv        (per subject, per channel)
    - fooof_AllROIs.csv            (per subject, per ROI and global)
    - fooof_FinalWide_perSubject.csv (one row per subject, wide format)


Dependencies
------------

  pip install pandas numpy


Script Functionality
--------------------

For each subject:

  - Loads all FOOOF features from *_features.csv
  - Assigns each channel to a Region of Interest (ROI) using channel name prefixes
  - For each frequency band (delta, theta, alpha, beta, gamma):
      - Selects the peak with the **highest Power (PW)** in the band range
      - Extracts Center Frequency (CF), Power (PW), Bandwidth (BW)
  - Appends global FOOOF parameters:
      - aperiodic_offset
      - aperiodic_exponent
      - model fit error and R²
      - total power (dB and linear) per band
  - Averages all features per ROI and globally
  - Compiles all results into:
      - Channel-level DataFrame (long format)
      - ROI-level DataFrame (with Global row)
      - Subject-level wide-format summary


Frequency Bands Used
--------------------

  - Delta: 1–4 Hz
  - Theta: 4–8 Hz
  - Alpha: 8–13 Hz
  - Beta: 13–30 Hz
  - Gamma: 30–45 Hz


ROI Mapping
-----------

Channels are assigned to simplified ROIs based on name prefixes:

  - Prefrontal: Fp, AF
  - Frontal:    F
  - Central:    C, FC, CP
  - Parietal:   P, PO
  - Temporal:   T, TP, FT
  - Occipital:  O, Iz


Output Files
------------

  - fooof_AllChannels.csv
      One row per subject × channel
      Includes spectral peaks and power metrics for each band

  - fooof_AllROIs.csv
      One row per subject × ROI
      Includes averaged features across channels in each ROI
      Includes one Global row per subject

  - fooof_FinalWide_perSubject.csv
      One row per subject
      Wide format: all channel and ROI metrics expanded into single row


Final Output Summary
---------------------

After running the script, you will have:

  - A long-format table for per-channel and per-ROI features
  - A subject-level table with all features ready for statistical analysis or modeling
  - Clean and consistent feature tables for downstream analysis

------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------

Script: 5_2_PuttingFeaturesTogether_Connectivity.py
===================================================

Aggregating Functional Connectivity Features Across Subjects
------------------------------------------------------------

This script processes subject-level weighted Phase Lag Index (wPLI) matrices
(from previous connectivity analysis) and produces:

  1. A full long-format table of all subject × ROI × ROI × frequency band values
  2. A wide-format feature table for machine learning or statistical modeling


Input and Output Paths
-----------------------

Input: Subject-level wPLI matrices (per frequency band)
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Connectivity_wPLI_bands\subID\wpli_<band>.csv

Output: Aggregated tables
  Path: P:\<HOSPITAL_NETWORK>\
        <BASE_DIR>\Preprocessed\Features\

    - all_subjects_wpli_longformat.csv   (long-format: one row per connection)
    - wpli_features_wide.csv             (wide-format: one row per subject)


Dependencies
------------

  pip install pandas tqdm


Script Functionality
--------------------

For each subject:

  - Loads wPLI matrices for the 5 standard EEG frequency bands
  - For the **long-format** table:
      - Extracts every connection (ROI1 × ROI2) for every band
      - Records one row per subject, per ROI pair, per frequency band
  - For the **wide-format** feature matrix:
      - Uses only the **lower triangle** of the wPLI matrix (to avoid redundancy)
      - Each connection becomes a feature column (e.g., `wpli_alpha_PFC_PPC`)
      - Compiles one row per subject with all pairwise connectivity values

The outputs are suitable for network analysis, statistical comparisons, and machine learning.


Frequency Bands Used
--------------------

  - Delta: 1–4 Hz
  - Theta: 4–8 Hz
  - Alpha: 8–13 Hz
  - Beta: 13–30 Hz
  - Gamma: 30–45 Hz


Output Files
------------

  - all_subjects_wpli_longformat.csv
      One row per subject × band × ROI pair
      Useful for visualizations, heatmaps, or statistical summaries

  - wpli_features_wide.csv
      One row per subject
      Each column is a unique ROI pair × frequency band
      Suitable for feature-based modeling


Final Output Summary
---------------------

After running the script, you will have:

  - A long-format dataset with complete wPLI values across ROI pairs and bands
  - A wide-format table ready for statistical analysis or classification tasks
  - Clean and organized connectivity features for downstream research



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------


Script: 5_3_PuttingFeaturesTogether_wplifooof.py
================================================

Combining Spectral and Connectivity Features into Final Dataset
---------------------------------------------------------------

This script merges the wide-format feature tables from:

  1. Spectral analysis (FOOOF-derived features)
  2. Functional connectivity (wPLI features)

...into a single, clean dataset for modeling or statistical analysis.
It also generates a log file summarizing the merged feature space.


Input and Output Paths
-----------------------

Input:
  - Spectral features (FOOOF):
      Path: P:\...\Preprocessed\Features\fooof_FinalWide_perSubject.csv
  - Connectivity features (wPLI):
      Path: P:\...\Preprocessed\Features\wpli_features_wide.csv

Output:
  - Combined feature matrix:
      Path: P:\...\Preprocessed\Features\combined_features_wide.csv

  - Summary log:
      Path: P:\...\Preprocessed\Features\combined_features_summary_log.txt


Dependencies
------------

  pip install pandas


Script Functionality
--------------------

  - Loads FOOOF and wPLI feature tables (one row per subject)
  - Merges them using `subject_id` as the key
  - Saves the final combined feature matrix as a CSV
  - Generates a human-readable summary log with:
      - Number of subjects and features
      - Data type breakdown
      - Percentage of missing values per column (if any)
      - Estimated memory usage


Output Files
------------

  - combined_features_wide.csv
      One row per subject
      All spectral and connectivity features combined

  - combined_features_summary_log.txt
      Text log with dataset overview, missing value summary, and memory stats


Final Output Summary
---------------------

After running this script, you will have:

  - A full dataset ready for statistical testing, machine learning, or export
  - A concise summary describing the structure and integrity of the combined features



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------


Script: 5_4_Adding subject data.py
==================================

Merging Subject Metadata (Demographics & Group Info) into Feature Matrix
------------------------------------------------------------------------

This script joins the final combined feature set (spectral + connectivity)
with demographic and group-level information from the BIDS `participants.tsv` file.

The output is a fully annotated feature matrix for each subject, containing:

  - EEG-derived features (from previous scripts)
  - Subject metadata (e.g., age, sex, group, twin info, clinical label)


Input and Output Paths
-----------------------

Input: Combined feature matrix
  Path: P:\...\Preprocessed\Features\combined_features_wide.csv

Input: BIDS-compliant participant metadata
  Path: P:\...\BIDS\participants.tsv

Output: Final annotated dataset
  Path: P:\...\Preprocessed\Features\combined_features_wide_SubjectInfo.csv


Dependencies
------------

  pip install pandas


Script Functionality
--------------------

  - Loads the combined EEG feature set (spectral + connectivity)
  - Loads participant metadata from `participants.tsv`
  - Cleans and normalizes `subject_id` fields:
      - Removes 'sub-' prefix
      - Ensures IDs are 7-character zero-padded strings
  - Checks for ID mismatches:
      - Reports subjects found only in features or only in participants file
  - Merges feature and metadata tables on `subject_id`
  - Reorders columns so demographic info appears before feature columns
  - Saves merged dataset as CSV


Metadata Columns Added
-----------------------

From `participants.tsv`, the following fields are merged into the feature table:

  - participant_id
  - sex
  - age
  - group
  - pair_id
  - twin_type_ab
  - clinical_group (e.g., Pr0, HP1, HC2)


Output Files
------------

  - combined_features_wide_SubjectInfo.csv
      One row per subject
      Includes all features and demographic annotations
      Ready for statistical modeling and group analysis


Final Output Summary
---------------------

After running the script, you will have:

  - A fully annotated dataset with all extracted EEG features
  - Consistent subject IDs across features and metadata
  - A clean CSV ready for final analysis, ML models, or visualization



------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------
------------------------------------------------

