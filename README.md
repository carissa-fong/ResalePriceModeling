# Resale Price Modeling

This repository contains the data-preparation pipeline, exploratory analysis, and final modeling artifacts for the HDB resale price study.

The project is organized around two main stages:

- a reproducible data pipeline that transforms the raw resale transactions file into the final modeling dataset
- a results layer that stores the final linear-regression and decision-tree modeling outputs

## Repository Structure

### Active project files

- [01_data_preprocessing_pipeline.ipynb](01_data_preprocessing_pipeline.ipynb)  
  Base cleaning and preprocessing of the raw resale transactions data.

- [02_feature_enrichment_pipeline.ipynb](02_feature_enrichment_pipeline.ipynb)  
  Feature engineering and enrichment pipeline that produces the final modeling dataset.

- [Resale_price_data_analysis.ipynb](Resale_price_data_analysis.ipynb)  
  Exploratory analysis notebook used for descriptive inspection of the resale price data.

### Data folder

- `data/Resale Flat Prices from Jan 2015 to Feb 2026.csv`  
  Required raw input file.

- `data/hdb_resale_pipeline_intermediate.csv`  
  Intermediate output from Step 1 of the preprocessing workflow.

- `data/hdb_resale_final_modeling_dataset_2015_2025.csv`  
  Final modeling dataset used by the current decision-tree workflow.

Other files in `data/` are supporting enrichment sources or intermediate merge artifacts retained for reproducibility.

### Results folder

- [results/DT_models](results/DT_models)  
  Final decision-tree workflow and outputs. This includes:
  - `01_step1_naive_tree_baseline.ipynb`
  - `02_step2_regularized_teacher_tree.ipynb`
  - `03_step3_teacher_tree_interpretation.ipynb`
  - `03_step3_teacher_tree_summary.json`
  - `02_step2_teacher_tree.txt`
  - `figures/` containing the baseline tree, teacher tree, feature-importance plot, and zoomed branch figures

- [results/LR_models](results/LR_models)  
  Final linear-regression materials and write-up artifacts. This currently includes:
  - `Linear regression code.Rmd`
  - `RMSE and MAE test for changed factors.xlsx`

## Data Pipeline

The current dataset-generation workflow consists of two notebooks:

### Step 1: Base preprocessing

[01_data_preprocessing_pipeline.ipynb](01_data_preprocessing_pipeline.ipynb)

This notebook performs the initial cleaning and transformation of the raw HDB resale transactions data and writes:

- `data/hdb_resale_pipeline_intermediate.csv`

### Step 2: Feature enrichment

[02_feature_enrichment_pipeline.ipynb](02_feature_enrichment_pipeline.ipynb)

This notebook enriches the intermediate dataset with the project’s engineered features, filters the study window to `2015-01` through `2025-12`, and writes:

- `data/hdb_resale_final_modeling_dataset_2015_2025.csv`

## Required Input

Place the raw resale transactions file in:

- `data/Resale Flat Prices from Jan 2015 to Feb 2026.csv`

The pipeline expects that filename exactly.

## How To Rebuild The Dataset

From the repository root, run:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_data_preprocessing_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_feature_enrichment_pipeline.ipynb
```

If both notebooks complete successfully, the main output will be:

- `data/hdb_resale_final_modeling_dataset_2015_2025.csv`

## Modeling Workflows

The repository keeps two final modeling tracks: a linear-regression workflow and a decision-tree workflow. Both are built from the same enriched dataset and are intended to provide complementary perspectives on resale price behavior.

### Linear Regression Workflow

The linear-regression materials are stored in [results/LR_models](results/LR_models), with the main modeling write-up in [Linear regression code.Rmd](results/LR_models/Linear%20regression%20code.Rmd).

The linear-regression workflow is organized in two parts:

- a structural baseline model using `floor_area_sqm`, `remaining_lease_years`, `storey_mid`, `flat_type`, and `flat_model`
- a refined log-linear final model fitted on `log_resale_price` and simplified through backward stepwise AIC selection

The baseline is evaluated using Adjusted `R²`, exact-dollar RMSE, and Mean Bias Error (MBE). The final log-linear model retains a smaller set of key explanatory variables, including:

- `mature_estate`
- `floor_area_sqm`
- `storey_mid`
- `remaining_lease_years`
- `dist_to_cbd_m`
- `sora_monthly_avg`
- `Per_Capita_GDP`

Because the dependent variable is log-transformed, coefficient interpretation follows the standard percentage-change transformation:

- `(exp(beta) - 1) * 100`

This allows the final regression model to be discussed in substantive terms such as price premiums for mature estates and marginal percentage changes associated with floor area, lease length, storey, distance to CBD, and macroeconomic conditions.

### Decision-Tree Workflow

The final decision-tree workflow is stored in [results/DT_models](results/DT_models).

This workflow uses:

- a shallow CART baseline as Step 1
- a regularized teacher-tree search as Step 2
- a teacher-tree interpretation notebook as Step 3

The current tracked specification uses the following numeric features:

- `storey_mid`
- `remaining_lease_years`
- `mrt_walking_distance`
- `dist_to_cbd_m`
- `dist_nearest_park_m`
- `dist_nearest_hawker_centre_m`
- `sora_monthly_avg`

and the following categorical features:

- `region`
- `mature_estate`
- `flat_type`
- `flat_model`

This tracked decision-tree specification is an interpretability-oriented setup. In the final saved workflow, `floor_area_sqm`, `region_URA`, and `Per_Capita_GDP` are not used in the active tree so that the reported split structure emphasizes housing type, lease, financing, and location effects more clearly.

The main active outputs are:

- [results/DT_models/03_step3_teacher_tree_summary.json](results/DT_models/03_step3_teacher_tree_summary.json)
- [results/DT_models/figures/02_step2_teacher_tree.png](results/DT_models/figures/02_step2_teacher_tree.png)
- [results/DT_models/02_step2_teacher_tree.txt](results/DT_models/02_step2_teacher_tree.txt)

## Final Dataset Schema

The final modeling dataset contains these columns:

- `region`
- `region_URA`
- `mature_estate`
- `flat_type`
- `flat_model`
- `floor_area_sqm`
- `storey_mid`
- `remaining_lease_years`
- `mrt_walking_distance`
- `dist_to_cbd_m`
- `dist_nearest_park_m`
- `dist_nearest_hawker_centre_m`
- `sora_monthly_avg`
- `Per_Capita_GDP`
- `resale_price`
- `log_resale_price`

## Notes

- The intermediate dataset is an expected part of the pipeline.
- The final modeling dataset is the main input for both the linear-regression and decision-tree workflows.
- Some enrichment fields may remain missing if optional supporting data is unavailable locally, but the pipeline is designed to complete as long as the required sources are present.
- The active final model folders are `results/LR_models` and `results/DT_models`; historical or exploratory materials are kept outside those tracked result folders.
