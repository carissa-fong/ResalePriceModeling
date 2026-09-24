from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd


def find_project_root() -> Path:
    module_path = Path(__file__).resolve()
    for candidate in [module_path.parent, *module_path.parents]:
        if (candidate / "data" / "hdb_resale_final_modeling_dataset_2015_2025.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find project root.")


PROJECT_ROOT = find_project_root()
DT_V4_DIR = Path(__file__).resolve().parent
if str(DT_V4_DIR) not in sys.path:
    sys.path.insert(0, str(DT_V4_DIR))

from dt_modelsv4_utils import JSON_PATH  # noqa: E402
from generate_v4_artifacts import run_analysis, write_outputs  # noqa: E402


def ensure_summary() -> dict:
    if not JSON_PATH.exists():
        summary = run_analysis()
        write_outputs(summary)
        return summary
    summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if "step3_teacher_interpretation" not in summary:
        summary = run_analysis()
        write_outputs(summary)
    return summary


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


SETUP_CELL = """
from pathlib import Path
import sys

def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "hdb_resale_final_modeling_dataset_2015_2025.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root from the current working directory.")

PROJECT_ROOT = find_project_root()
DT_V4_DIR = PROJECT_ROOT / "results" / "DT_models"
if str(DT_V4_DIR) not in sys.path:
    sys.path.insert(0, str(DT_V4_DIR))
"""


def build_step1_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Step 1: Naive Decision Tree Baseline

            This notebook establishes the shallow CART baseline used to benchmark the final regularized teacher tree.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            import json
            import pandas as pd
            from IPython.display import Image, display

            from dt_modelsv4_utils import (
                CATEGORICAL_FEATURES,
                JSON_PATH,
                NUMERIC_FEATURES,
                fit_naive_baseline,
                load_complete_case_dataset,
                split_dataset,
            )

            df, data_info = load_complete_case_dataset()
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
            baseline_results = fit_naive_baseline(X_train, y_train, X_val, y_val, X_test, y_test)

            print(data_info)
            print("Numeric features:", NUMERIC_FEATURES)
            print("Categorical features:", CATEGORICAL_FEATURES)

            baseline_table = pd.DataFrame(
                [
                    {"split": split_name, **metrics}
                    for split_name, metrics in baseline_results.items()
                ]
            )
            display(baseline_table.round(4))

            summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))
            display(Image(filename=summary["artifacts"]["baseline_png_path"]))
            """
        ),
    ]
    return nb


def build_step2_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Step 2: Regularized Teacher Tree Selection

            This notebook searches a small grid of regularized CART teacher trees and selects the simplest model whose validation RMSE remains near-optimal.

            The selected teacher is the final predictive and explanatory model for the final decision-tree workflow.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            import json
            import pandas as pd
            from IPython.display import Image, display
            from dt_modelsv4_utils import JSON_PATH

            summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))

            teacher_grid = pd.DataFrame(
                [
                    {
                        "max_depth": row["config"]["max_depth"],
                        "min_samples_leaf": row["config"]["min_samples_leaf"],
                        "min_samples_split": row["config"]["min_samples_split"],
                        "depth": row["depth"],
                        "leaf_count": row["leaf_count"],
                        "validation_rmse": row["validation"]["rmse"],
                        "validation_r2": row["validation"]["r2"],
                        "test_rmse": row["test"]["rmse"],
                        "test_r2": row["test"]["r2"],
                    }
                    for row in summary["step2_teacher_grid"]
                ]
            )
            display(teacher_grid.round(4))

            teacher_importance = pd.DataFrame(summary["step2_selected_teacher"]["feature_importances"])
            display(teacher_importance.head(12).round(4))

            print("Selected teacher root:", summary["step2_selected_teacher"]["root_summary"])

            display(Image(filename=summary["artifacts"]["teacher_importance_png_path"]))
            display(Image(filename=summary["artifacts"]["teacher_png_path"]))
            """
        ),
    ]
    return nb


def build_step3_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Step 3: Teacher Tree Interpretation

            The aim is to understand the upper-level split structure, the most frequently used features, and the largest leaf rules with sufficient sample support.
            """
        ),
        code(SETUP_CELL),
        code(
            """
            import json
            import pandas as pd
            from IPython.display import Image, Markdown, display
            from dt_modelsv4_utils import JSON_PATH

            summary = json.loads(JSON_PATH.read_text(encoding="utf-8"))

            interpretation = summary["step3_teacher_interpretation"]

            display(Markdown("## Tree Overview"))
            overview = pd.DataFrame(
                [
                    {
                        "depth": interpretation["depth"],
                        "leaf_count": interpretation["leaf_count"],
                        "root_rule": interpretation["root_summary"]["rule"],
                    }
                ]
            )
            display(overview)

            split_usage = pd.DataFrame(
                [
                    {"feature": feature, "internal_node_count": count}
                    for feature, count in interpretation["split_feature_usage"].items()
                ]
            ).sort_values(["internal_node_count", "feature"], ascending=[False, True])
            display(Markdown("## Most-Used Split Features"))
            display(split_usage.head(10))

            top_leaf_rules = pd.DataFrame(interpretation["top_leaf_rules_by_samples"])
            if not top_leaf_rules.empty:
                top_leaf_rules = top_leaf_rules.loc[:, ["samples", "prediction", "path"]].copy()
                top_leaf_rules["prediction"] = top_leaf_rules["prediction"].round(2)
            display(Markdown("## Largest Leaf Rules"))
            display(top_leaf_rules)

            display(Markdown("## Teacher Tree Figure"))
            display(Image(filename=summary["artifacts"]["teacher_png_path"]))
            """
        ),
    ]
    return nb


def write_notebook(path: Path, notebook: nbf.NotebookNode) -> None:
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbf.write(notebook, path)


def main() -> None:
    write_notebook(DT_V4_DIR / "01_step1_naive_tree_baseline.ipynb", build_step1_notebook())
    write_notebook(DT_V4_DIR / "02_step2_regularized_teacher_tree.ipynb", build_step2_notebook())
    write_notebook(DT_V4_DIR / "03_step3_teacher_tree_interpretation.ipynb", build_step3_notebook())
    print("Wrote final decision-tree notebook pipeline.")


if __name__ == "__main__":
    main()
