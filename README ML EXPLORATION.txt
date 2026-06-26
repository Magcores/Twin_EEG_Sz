Machine Learning Analysis Summary
----------------------------------

Group Comparisons and Sample Sizes
----------------------------------

The following pairwise group comparisons were included in the analysis:

- Pr0 vs HP1 (Pr0_minus_HP1, HP1_minus_Pr0)
- Original HC2 vs Original HP1
- Original HC2 vs Original Pr0
- HCa vs HCb (HCa_minus_HCb, HCb_minus_HCa)

Sample sizes per label:

- Original_HC2 — 68 subjects
- Original_HP1 — 49 subjects
- Original_Pr0 — 44 subjects
- Pr0_minus_HP1 — 43 subjects
- HP1_minus_Pr0 — 43 subjects
- HCa_minus_HCb — 30 subjects
- HCb_minus_HCa — 30 subjects

Note:
The lower sample sizes in paired comparisons (e.g., Pr0_minus_HP1, HCa_minus_HCb — especially HC twins) are due to unmatched or missing twin codes. Some twins could not be paired and were excluded.

Feature Processing: Standardization & PCA
-----------------------------------------

Scaling:
- Features were standardized to zero mean and unit variance using StandardScaler.

Dimensionality Reduction:
- Principal Component Analysis (PCA) was used to retain enough components to explain 95% of the variance.

PCA Summary:
- Twin-difference features were standardized (mean ≈ 0, std ≈ 1)
- PCA was completed with the following results:
  - Original feature dimensions: 11,844
  - Reduced dimensions: 117
  - Total explained variance: 95.14%

Explained Variance per PC:
- PC1: 11.40%
- PC2: 3.19%
- PC3: 2.92%
- PC4: 2.53%
- PC5: 2.29%
- PC6: 2.19%
- PC7: 2.15%
- PC8: 2.02%
- PC9: 1.84%
- PC10: 1.67%

Top 10 Features Contributing to Principal Components:
- PC1: EEG power features (e.g., gamma_dB_AF4, beta_dB_F7)
- PC2 & PC3: Functional connectivity measures (e.g., wpli_delta_precentral-rh_insula-rh, wpli_theta_precuneus-rh_frontalpole-rh)

Nested Cross-Validation for Classification
-------------------------------------------

The outer loop estimates generalization performance, and the inner loop tunes feature selection:

- Outer CV: 5-fold StratifiedKFold
- Inner CV: 3-fold StratifiedKFold for feature selection

Feature Selection:
- Sequential Forward Selection (up to 10 features) using logistic regression accuracy as the scoring metric.

Models Tested:
- Support Vector Classifier (RBF kernel; C = 0.01, 0.1, 1)
- Random Forest (max depth = 5, 10, 25)
- Gradient Boosting (max depth = 5, 10, 25)

Metrics Recorded:
- Accuracy
- Precision
- Recall
- F1-score
(for both outer and inner CV folds)

Output Files:
- NestedCV_comparison_summary.csv

Visualizations:
- Multiple bar plot visualizations per model/parameter set
- Solid bars: Outer CV
- Hatched bars: Inner CV
- Separate “accuracy-only” plots are generated for clarity

Nested CV Results
------------------

Nested CV Results
------------------

Model               | Comparison                          | Accuracy (Outer) | Precision (Outer) | Recall (Outer) | F1-score (Outer) | Accuracy (Inner) | Precision (Inner) | Recall (Inner) | F1-score (Inner)
--------------------|--------------------------------------|------------------|-------------------|----------------|------------------|------------------|-------------------|----------------|------------------
SVC                 | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.510            | 0.322             | 0.533          | 0.389            | 0.599            | 0.611             | 0.596          | 0.586
SVC                 | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.510            | 0.322             | 0.533          | 0.389            | 0.662            | 0.666             | 0.663          | 0.661
SVC                 | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.604            | 0.609             | 0.604          | 0.598            | 0.654            | 0.656             | 0.654          | 0.652
Random Forest       | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.592            | 0.592             | 0.593          | 0.588            | 0.668            | 0.669             | 0.668          | 0.668
Random Forest       | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.627            | 0.628             | 0.628          | 0.626            | 0.668            | 0.669             | 0.668          | 0.667
Random Forest       | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.627            | 0.628             | 0.628          | 0.626            | 0.668            | 0.669             | 0.668          | 0.667
Gradient Boosting   | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.615            | 0.616             | 0.611          | 0.610            | 0.631            | 0.631             | 0.630          | 0.629
Gradient Boosting   | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.604            | 0.609             | 0.603          | 0.600            | 0.619            | 0.620             | 0.619          | 0.618
Gradient Boosting   | Pr0_minus_HP1 vs HP1_minus_Pr0       | 0.604            | 0.609             | 0.603          | 0.600            | 0.619            | 0.620             | 0.619          | 0.618
SVC                 | Original_HC2 vs Original_HP1         | 0.419            | 0.209             | 0.500          | 0.295            | 0.483            | 0.383             | 0.499          | 0.415
SVC                 | Original_HC2 vs Original_HP1         | 0.494            | 0.345             | 0.508          | 0.341            | 0.560            | 0.331             | 0.498          | 0.387
SVC                 | Original_HC2 vs Original_HP1         | 0.557            | 0.549             | 0.550          | 0.545            | 0.650            | 0.641             | 0.634          | 0.634
Random Forest       | Original_HC2 vs Original_HP1         | 0.566            | 0.537             | 0.534          | 0.521            | 0.637            | 0.625             | 0.611          | 0.610
Random Forest       | Original_HC2 vs Original_HP1         | 0.575            | 0.540             | 0.540          | 0.520            | 0.643            | 0.634             | 0.613          | 0.611
Random Forest       | Original_HC2 vs Original_HP1         | 0.583            | 0.536             | 0.550          | 0.530            | 0.643            | 0.634             | 0.613          | 0.611
Gradient Boosting   | Original_HC2 vs Original_HP1         | 0.549            | 0.526             | 0.523          | 0.521            | 0.568            | 0.551             | 0.548          | 0.547
Gradient Boosting   | Original_HC2 vs Original_HP1         | 0.557            | 0.544             | 0.547          | 0.542            | 0.564            | 0.557             | 0.557          | 0.555
Gradient Boosting   | Original_HC2 vs Original_HP1         | 0.540            | 0.515             | 0.521          | 0.515            | 0.564            | 0.557             | 0.557          | 0.555
SVC                 | Original_HC2 vs Original_Pr0         | 0.480            | 0.240             | 0.500          | 0.321            | 0.447            | 0.314             | 0.496          | 0.364
SVC                 | Original_HC2 vs Original_Pr0         | 0.607            | 0.304             | 0.500          | 0.378            | 0.553            | 0.386             | 0.504          | 0.415
SVC                 | Original_HC2 vs Original_Pr0         | 0.581            | 0.571             | 0.559          | 0.549            | 0.647            | 0.629             | 0.626          | 0.627
Random Forest       | Original_HC2 vs Original_Pr0         | 0.590            | 0.557             | 0.547          | 0.540            | 0.643            | 0.619             | 0.596          | 0.592
Random Forest       | Original_HC2 vs Original_Pr0         | 0.563            | 0.523             | 0.516          | 0.507            | 0.634            | 0.605             | 0.585          | 0.579
Random Forest       | Original_HC2 vs Original_Pr0         | 0.563            | 0.523             | 0.516          | 0.507            | 0.634            | 0.605             | 0.585          | 0.579
Gradient Boosting   | Original_HC2 vs Original_Pr0         | 0.581            | 0.568             | 0.553          | 0.546            | 0.623            | 0.601             | 0.596          | 0.596
Gradient Boosting   | Original_HC2 vs Original_Pr0         | 0.528            | 0.510             | 0.509          | 0.502            | 0.607            | 0.592             | 0.591          | 0.590
Gradient Boosting   | Original_HC2 vs Original_Pr0         | 0.528            | 0.510             | 0.509          | 0.502            | 0.607            | 0.591             | 0.590          | 0.589
SVC                 | HCa_minus_HCb vs HCb_minus_HCa       | 0.733            | 0.745             | 0.733          | 0.729            | 0.708            | 0.716             | 0.708          | 0.705
SVC                 | HCa_minus_HCb vs HCb_minus_HCa       | 0.733            | 0.745             | 0.733          | 0.729            | 0.708            | 0.716             | 0.708          | 0.705
SVC                 | HCa_minus_HCb vs HCb_minus_HCa       | 0.717            | 0.722             | 0.717          | 0.712            | 0.717            | 0.722             | 0.717          | 0.715
Random Forest       | HCa_minus_HCb vs HCb_minus_HCa       | 0.683            | 0.695             | 0.683          | 0.675            | 0.671            | 0.672             | 0.671          | 0.670
Random Forest       | HCa_minus_HCb vs HCb_minus_HCa       | 0.650            | 0.656             | 0.650          | 0.642            | 0.671            | 0.672             | 0.671          | 0.670
Random Forest       | HCa_minus_HCb vs HCb_minus_HCa       | 0.650            | 0.656             | 0.650          | 0.642            | 0.671            | 0.672             | 0.671          | 0.670
Gradient Boosting   | HCa_minus_HCb vs HCb_minus_HCa       | 0.633            | 0.639             | 0.633          | 0.628            | 0.621            | 0.623             | 0.621          | 0.618
Gradient Boosting   | HCa_minus_HCb vs HCb_minus_HCa       | 0.633            | 0.639             | 0.633          | 0.628            | 0.625            | 0.629             | 0.625          | 0.622
Gradient Boosting   | HCa_minus_HCb vs HCb_minus_HCa       | 0.633            | 0.639             | 0.633          | 0.628            | 0.625            | 0.629             | 0.625          | 0.622


Note:
The classification accuracies are generally modest across comparisons, with
the exception of the HCa_minus_HCb vs HCb_minus_HCa contrast, which showed
unexpectedly high accuracy. However, when testing different random seeds, this
result was not consistent — suggesting that the higher performance may be
driven by the low sample size (would not explain the higher accuracy in the other layer, though). 
Caution is advised when interpreting findings from this comparison.

Overall, accuracies remained low regardless of using all EEG power or all
connectivity features. Using only region-of-interest (ROI) data or specific
EEG power features (e.g., theta peaks; gamma or beta db power) may lead to 
more informative and interpretable results.
