# Decision Tree Methodology

## Methodological Objective

This alternative decision-tree specification was designed to preserve the same regularized teacher-tree workflow while intentionally excluding `floor_area_sqm` and `region_URA`. The purpose of this alternative setup is to surface secondary drivers of resale price that may be masked when direct size and overlapping regional indicators dominate the split structure.

## Data Preparation

The analysis used `resale_price` as the response variable. Complete-case filtering was applied before model training so that all observations used in the analysis had non-missing values for every selected predictor and for the response variable. The dataset was then split into training, validation, and test subsets using a fixed random state. The training set was used to fit the trees, the validation set was used for model selection, and the test set was reserved for final evaluation.

## Feature Specification

The alternative explanatory feature set retained the following numeric predictors:

- `storey_mid`
- `remaining_lease_years`
- `mrt_walking_distance`
- `dist_to_cbd_m`
- `dist_nearest_park_m`
- `dist_nearest_hawker_centre_m`
- `sora_monthly_avg`

The categorical predictors were:

- `region`
- `mature_estate`
- `flat_type`
- `flat_model`

Categorical variables were one-hot encoded, while numeric variables were kept in their original scale.

The following variables were intentionally excluded from this alternative tree:

- `Per_Capita_GDP`, because it acts mainly as a macroeconomic regime proxy and tends to encode broad time effects rather than transaction-level housing structure.
- `floor_area_sqm`, to reduce the dominance of direct dwelling-size effects and allow secondary structural and financing variables to emerge more clearly.
- `region_URA`, to avoid overlapping broad geography signals and rely on the remaining location indicators more parsimoniously.

## Step 1: Naive Baseline

As a benchmark, a shallow CART baseline was first trained using the alternative feature set after preprocessing. This baseline used `max_depth = 4` and served as a reference point for evaluating whether the regularized teacher tree offered meaningful gains in predictive performance.

## Step 2: Regularized Teacher Tree Search

The main model was a regularized teacher tree trained using `DecisionTreeRegressor` on the encoded feature matrix. A compact hyperparameter grid was explored:

- `max_depth` in {5, 6}
- `min_samples_leaf` in {1000, 2000}
- `min_samples_split` in {200, 500}

Each candidate tree was evaluated using validation RMSE and R-squared, while train and test metrics were also recorded for context. Feature importances were aggregated back to their original source variables so that one-hot encoded categories could still be interpreted at the variable level. The selected teacher tree was chosen using a near-best validation rule: among all models within `1%` of the best validation RMSE, the simpler regularized configuration was preferred.

## Step 3: Teacher Tree Interpretation

The selected teacher tree is interpreted directly as the final report-facing tree. Interpretation focuses on the upper and middle levels of the tree, the most frequently used split features, and the highest-support leaf rules. This makes the tree suitable for report-level explanation while preserving the predictive structure learned by the regularized model.

## Visualisation

The teacher tree, its feature-importance plot, and the branch zoom-ins are exported to a dedicated `figures` folder so that all report visuals are stored separately from the code and notebook files. This keeps the workflow organised while making it easier to reference the final figures directly in the report.

## Final Methodological Position

Overall, this alternative pipeline can be described as a curated-feature, regularized CART workflow that removes direct floor-area and redundant regional signals in order to highlight secondary pricing structure. The resulting teacher tree is intended as a complementary interpretation to the main final model rather than a replacement for it.
