from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text


RANDOM_STATE = 42
VAL_TEST_SIZE = 2 / 11
TEACHER_SELECTION_TOLERANCE = 0.01
NUMERIC_FEATURES = [
    "storey_mid",
    "remaining_lease_years",
    "mrt_walking_distance",
    "dist_to_cbd_m",
    "dist_nearest_park_m",
    "dist_nearest_hawker_centre_m",
    "sora_monthly_avg",
]

CATEGORICAL_FEATURES = [
    "region",
    "mature_estate",
    "flat_type",
    "flat_model",
]

TARGET = "resale_price"


def find_project_root() -> Path:
    search_roots = [Path.cwd(), *Path.cwd().parents]
    for candidate in search_roots:
        if (candidate / "data" / "hdb_resale_final_modeling_dataset_2015_2025.csv").exists():
            return candidate

    module_path = Path(__file__).resolve()
    for candidate in [module_path.parent, *module_path.parents]:
        if (candidate / "data" / "hdb_resale_final_modeling_dataset_2015_2025.csv").exists():
            return candidate

    raise FileNotFoundError("Could not find the project root from the current working directory.")


PROJECT_ROOT = find_project_root()
RESULTS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = RESULTS_DIR / "figures"
BRANCH_FIGURES_DIR = FIGURES_DIR / "teacher_branches"
MPLCONFIGDIR = RESULTS_DIR / ".mplconfig"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
BRANCH_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

DATA_PATH = PROJECT_ROOT / "data" / "hdb_resale_final_modeling_dataset_2015_2025.csv"
JSON_PATH = RESULTS_DIR / "03_step3_teacher_tree_summary.json"
BASELINE_DOT_PATH = FIGURES_DIR / "01_step1_naive_tree_baseline.dot"
BASELINE_PNG_PATH = FIGURES_DIR / "01_step1_naive_tree_baseline.png"
TEACHER_DOT_PATH = FIGURES_DIR / "02_step2_teacher_tree.dot"
TEACHER_PNG_PATH = FIGURES_DIR / "02_step2_teacher_tree.png"
TEACHER_TEXT_PATH = RESULTS_DIR / "02_step2_teacher_tree.txt"
IMPORTANCE_PNG_PATH = FIGURES_DIR / "02_step2_teacher_feature_importance.png"


def metric(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def drop_complete_cases(df: pd.DataFrame, required_columns: list[str]) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.dropna(subset=required_columns).reset_index(drop=True)
    return cleaned, before - len(cleaned)


def load_complete_case_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    df, rows_dropped = drop_complete_cases(df_raw, required_columns)
    info = {
        "rows_before_drop": int(len(df_raw)),
        "rows_dropped": int(rows_dropped),
        "rows_after_drop": int(len(df)),
    }
    return df, info


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_full = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full,
        y,
        test_size=VAL_TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def fit_naive_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    preprocessor = build_preprocessor()
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return {
        "train": metric(y_train, model.predict(X_train)),
        "validation": metric(y_val, model.predict(X_val)),
        "test": metric(y_test, model.predict(X_test)),
    }


def fit_naive_baseline_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = build_preprocessor()
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def recover_source_feature(transformed_name: str) -> tuple[str, str, str | None]:
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "", 1), "numeric", None

    encoded_name = transformed_name.replace("cat__", "", 1)
    matches = [
        feature
        for feature in CATEGORICAL_FEATURES
        if encoded_name == feature or encoded_name.startswith(feature + "_")
    ]
    source_feature = max(matches, key=len) if matches else encoded_name
    category = encoded_name[len(source_feature) + 1 :] if encoded_name.startswith(source_feature + "_") else None
    return source_feature, "categorical", category


def pretty_transformed_name(transformed_name: str) -> str:
    source_feature, feature_type, category = recover_source_feature(transformed_name)
    if feature_type == "numeric":
        return source_feature
    if category is None:
        return source_feature
    return f"{source_feature}={category}"


def aggregate_feature_importances(feature_names: list[str], importances: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transformed_name, importance in zip(feature_names, importances):
        source_feature, feature_type, category = recover_source_feature(transformed_name)
        rows.append(
            {
                "transformed_feature": transformed_name,
                "source_feature": source_feature,
                "feature_type": feature_type,
                "category": category,
                "importance": float(importance),
            }
        )

    grouped = (
        pd.DataFrame(rows)
        .groupby(["source_feature", "feature_type"], as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    return grouped.to_dict(orient="records")


def select_teacher(rows: list[dict]) -> dict:
    best_rmse = min(row["validation"]["rmse"] for row in rows)
    eligible = [row for row in rows if row["validation"]["rmse"] <= best_rmse * (1 + TEACHER_SELECTION_TOLERANCE)]
    eligible.sort(
        key=lambda row: (
            row["config"]["max_depth"],
            -row["config"]["min_samples_leaf"],
            -row["config"]["min_samples_split"],
            row["validation"]["rmse"],
        )
    )
    return eligible[0]


def sample_pruning_alphas(ccp_alphas: np.ndarray, max_candidates: int = 16) -> list[float]:
    unique = np.unique(np.asarray(ccp_alphas, dtype=float))
    unique = unique[unique >= 0]
    if len(unique) == 0:
        return [0.0]
    if len(unique) <= max_candidates:
        sampled = unique
    else:
        idx = np.linspace(0, len(unique) - 1, num=max_candidates, dtype=int)
        sampled = unique[idx]
    sampled = np.unique(np.concatenate([[0.0], sampled]))
    return [float(alpha) for alpha in sampled]


def format_threshold(feature: str, value: float) -> str:
    if feature == "sora_monthly_avg":
        return f"{value:.3f}"
    if feature.endswith("_distance") or feature.startswith("dist_") or feature.endswith("_m"):
        return f"{value:,.0f}"
    if feature in {"storey_mid", "remaining_lease_years"}:
        return f"{value:.1f}"
    return f"{value:.2f}"


def summarize_split(transformed_name: str, threshold: float) -> str:
    source_feature, feature_type, category = recover_source_feature(transformed_name)
    if feature_type == "numeric":
        return f"{source_feature} <= {format_threshold(source_feature, threshold)}"
    if category is None:
        return f"{source_feature} active"
    return f"{source_feature} = {category}"


def split_condition_and_edges(transformed_name: str, threshold: float) -> tuple[str, str, str]:
    source_feature, feature_type, category = recover_source_feature(transformed_name)
    if feature_type == "numeric" or category is None:
        return summarize_split(transformed_name, threshold), "Yes", "No"
    return f"{source_feature} = {category}", "No", "Yes"


def collect_split_feature_usage(tree_model: DecisionTreeRegressor, feature_names: list[str]) -> dict[str, int]:
    tree = tree_model.tree_
    counts: Counter[str] = Counter()
    for feature_idx in tree.feature:
        if feature_idx < 0:
            continue
        source_feature, _, _ = recover_source_feature(feature_names[feature_idx])
        counts[source_feature] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _condition_state() -> dict[str, Any]:
    return {
        "numeric": {},
        "categorical": defaultdict(lambda: {"eq": None, "neq": set()}),
    }


def _clone_state(state: dict[str, Any]) -> dict[str, Any]:
    numeric = {feature: bounds.copy() for feature, bounds in state["numeric"].items()}
    categorical = defaultdict(lambda: {"eq": None, "neq": set()})
    for feature, value in state["categorical"].items():
        categorical[feature] = {"eq": value["eq"], "neq": set(value["neq"])}
    return {"numeric": numeric, "categorical": categorical}


def _update_condition(state: dict[str, Any], transformed_name: str, threshold: float, go_left: bool) -> None:
    source_feature, feature_type, category = recover_source_feature(transformed_name)
    if feature_type == "numeric":
        bounds = state["numeric"].setdefault(source_feature, {"lower": None, "upper": None})
        if go_left:
            bounds["upper"] = threshold if bounds["upper"] is None else min(bounds["upper"], threshold)
        else:
            bounds["lower"] = threshold if bounds["lower"] is None else max(bounds["lower"], threshold)
        return

    if category is None:
        return

    current = state["categorical"][source_feature]
    if go_left:
        current["neq"].add(category)
    else:
        current["eq"] = category


def _render_conditions(state: dict[str, Any]) -> list[str]:
    conditions: list[str] = []

    for feature, bounds in state["numeric"].items():
        lower = bounds["lower"]
        upper = bounds["upper"]
        if lower is not None and upper is not None:
            conditions.append(
                f"{feature} in ({format_threshold(feature, lower)}, {format_threshold(feature, upper)}]"
            )
        elif lower is not None:
            conditions.append(f"{feature} > {format_threshold(feature, lower)}")
        elif upper is not None:
            conditions.append(f"{feature} <= {format_threshold(feature, upper)}")

    for feature, info in state["categorical"].items():
        if info["eq"] is not None:
            conditions.append(f"{feature} = {info['eq']}")
        elif info["neq"]:
            if len(info["neq"]) <= 2:
                disallowed = ", ".join(sorted(info["neq"]))
                conditions.append(f"{feature} not in {{{disallowed}}}")

    return conditions


def node_rule_paths(tree_model: DecisionTreeRegressor, feature_names: list[str]) -> dict[int, str]:
    tree = tree_model.tree_
    paths: dict[int, str] = {}

    def walk(node_id: int, state: dict[str, Any]) -> None:
        conditions = _render_conditions(state)
        paths[int(node_id)] = " AND ".join(conditions) if conditions else "Root"

        feature_idx = tree.feature[node_id]
        if feature_idx < 0:
            return

        transformed_name = feature_names[feature_idx]
        threshold = float(tree.threshold[node_id])

        left_state = _clone_state(state)
        _update_condition(left_state, transformed_name, threshold, go_left=True)
        walk(tree.children_left[node_id], left_state)

        right_state = _clone_state(state)
        _update_condition(right_state, transformed_name, threshold, go_left=False)
        walk(tree.children_right[node_id], right_state)

    walk(0, _condition_state())
    return paths


def extract_leaf_rules(tree_model: DecisionTreeRegressor, feature_names: list[str]) -> list[dict[str, Any]]:
    tree = tree_model.tree_
    rules: list[dict[str, Any]] = []

    def walk(node_id: int, state: dict[str, Any]) -> None:
        feature_idx = tree.feature[node_id]
        if feature_idx < 0:
            conditions = _render_conditions(state)
            rules.append(
                {
                    "node_id": int(node_id),
                    "path": " AND ".join(conditions) if conditions else "Root",
                    "samples": int(tree.n_node_samples[node_id]),
                    "prediction": float(tree.value[node_id][0, 0]),
                }
            )
            return

        transformed_name = feature_names[feature_idx]
        threshold = float(tree.threshold[node_id])

        left_state = _clone_state(state)
        _update_condition(left_state, transformed_name, threshold, go_left=True)
        walk(tree.children_left[node_id], left_state)

        right_state = _clone_state(state)
        _update_condition(right_state, transformed_name, threshold, go_left=False)
        walk(tree.children_right[node_id], right_state)

    walk(0, _condition_state())
    return rules


def root_split_summary(tree_model: DecisionTreeRegressor, feature_names: list[str]) -> dict[str, Any] | None:
    tree = tree_model.tree_
    root_feature_idx = tree.feature[0]
    if root_feature_idx < 0:
        return None

    feature_name = feature_names[root_feature_idx]
    threshold = float(tree.threshold[0])
    left_id = tree.children_left[0]
    right_id = tree.children_right[0]
    condition, left_split, right_split = split_condition_and_edges(feature_name, threshold)

    return {
        "feature": pretty_transformed_name(feature_name),
        "threshold": threshold,
        "rule": condition,
        "left_child_samples": int(tree.n_node_samples[left_id]) if left_id != -1 else None,
        "right_child_samples": int(tree.n_node_samples[right_id]) if right_id != -1 else None,
        "left_branch": left_split,
        "right_branch": right_split,
    }


def _escape_dot(text: str) -> str:
    return text.replace('"', '\\"')


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _interpolate_color(value: float, value_min: float, value_max: float) -> str:
    low = _hex_to_rgb("#fff7ec")
    high = _hex_to_rgb("#e58139")
    if value_max <= value_min:
        return _rgb_to_hex(low)
    ratio = max(0.0, min(1.0, (value - value_min) / (value_max - value_min)))
    rgb = tuple(int(round(low[i] + ratio * (high[i] - low[i]))) for i in range(3))
    return _rgb_to_hex(rgb)


def export_tree_dot(
    tree_model: DecisionTreeRegressor,
    feature_names: list[str],
    root_node: int = 0,
    title: str | None = None,
    presentation: str = "original",
) -> str:
    tree = tree_model.tree_
    all_values = tree.value[:, 0, 0]
    value_min = float(np.min(all_values))
    value_max = float(np.max(all_values))

    lines = [
        "digraph Tree {",
        'graph [rankdir=TB, splines=true, nodesep=0.45, ranksep=0.7];',
        'node [shape=box, style="rounded,filled", color="#2f3e46", fontname="Helvetica", fontsize=10, margin="0.14,0.10"];',
        'edge [color="#4f5d75", fontname="Helvetica", fontsize=9];',
    ]
    if title:
        lines.append(f'labelloc="t";')
        lines.append(f'label="{_escape_dot(title)}";')

    visited: set[int] = set()

    def walk(node_id: int) -> None:
        if node_id in visited or node_id == -1:
            return
        visited.add(node_id)

        feature_idx = tree.feature[node_id]
        samples = int(tree.n_node_samples[node_id])
        value = float(tree.value[node_id][0, 0])
        fillcolor = _interpolate_color(value, value_min, value_max)

        if presentation == "original":
            if feature_idx < 0:
                label = (
                    f"samples = {samples:,}\\n"
                    f"value = {value:,.0f}"
                )
                lines.append(f'N{node_id} [label="{_escape_dot(label)}", fillcolor="{fillcolor}"];')
                return

            transformed_name = feature_names[feature_idx]
            threshold = float(tree.threshold[node_id])
            condition = summarize_split(transformed_name, threshold)
            label = (
                f"{condition}\\n"
                f"samples = {samples:,}\\n"
                f"value = {value:,.0f}"
            )
            lines.append(f'N{node_id} [label="{_escape_dot(label)}", fillcolor="{fillcolor}"];')

            left_id = int(tree.children_left[node_id])
            right_id = int(tree.children_right[node_id])
            walk(left_id)
            walk(right_id)
            lines.append(f"N{node_id} -> N{left_id};")
            lines.append(f"N{node_id} -> N{right_id};")
            return

        if feature_idx < 0:
            label = f"Leaf\\nn={samples:,}\\nmean=${value:,.0f}"
            lines.append(f'N{node_id} [label="{_escape_dot(label)}", fillcolor="{fillcolor}"];')
            return

        transformed_name = feature_names[feature_idx]
        threshold = float(tree.threshold[node_id])
        condition, left_edge, right_edge = split_condition_and_edges(transformed_name, threshold)
        label = f"{condition}\\nn={samples:,}\\nmean=${value:,.0f}"
        lines.append(f'N{node_id} [label="{_escape_dot(label)}", fillcolor="{fillcolor}"];')

        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])
        walk(left_id)
        walk(right_id)
        lines.append(f'N{node_id} -> N{left_id} [label="{left_edge}"];')
        lines.append(f'N{node_id} -> N{right_id} [label="{right_edge}"];')

    walk(int(root_node))
    lines.append("}")
    return "\n".join(lines)


def export_tree_text_report(tree_model: DecisionTreeRegressor, feature_names: list[str]) -> str:
    pretty_feature_names = [pretty_transformed_name(name) for name in feature_names]
    tree_text = export_text(
        tree_model,
        feature_names=pretty_feature_names,
        decimals=1,
        show_weights=True,
    )
    return tree_text.rstrip() + "\n"


def build_interesting_teacher_branches(
    tree_model: DecisionTreeRegressor,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    tree = tree_model.tree_
    root_left = int(tree.children_left[0])
    root_right = int(tree.children_right[0])
    path_map = node_rule_paths(tree_model, feature_names)
    branches: list[dict[str, Any]] = []

    if root_left != -1:
        branches.append(
            {
                "name": "non_three_room_branch",
                "title": "Non-3-Room Branch",
                "node_id": root_left,
                "path": path_map[root_left],
            }
        )

    if root_right != -1:
        branches.append(
            {
                "name": "three_room_branch",
                "title": "3-Room Branch",
                "node_id": root_right,
                "path": path_map[root_right],
            }
        )

        right_left = int(tree.children_left[root_right])
        right_right = int(tree.children_right[root_right])

        if right_left != -1:
            branches.append(
                {
                    "name": "older_lease_three_room_branch",
                    "title": "Older-Lease 3-Room Branch",
                    "node_id": right_left,
                    "path": path_map[right_left],
                }
            )

        if right_right != -1:
            branches.append(
                {
                    "name": "longer_lease_three_room_branch",
                    "title": "Longer-Lease 3-Room Branch",
                    "node_id": right_right,
                    "path": path_map[right_right],
                }
            )

    return branches


def render_png_from_dot(
    dot_source: str,
    dot_path: Path = TEACHER_DOT_PATH,
    png_path: Path = TEACHER_PNG_PATH,
    unflatten_args: list[str] | None = None,
    dpi: int | None = None,
) -> bool:
    dot_path.write_text(dot_source, encoding="utf-8")
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        return False

    render_source = dot_path
    temp_path: Path | None = None

    if unflatten_args:
        unflatten_binary = shutil.which("unflatten")
        if unflatten_binary is not None:
            flattened = subprocess.run(
                [unflatten_binary, *unflatten_args, str(dot_path)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".dot",
                prefix=f"{dot_path.stem}_",
                delete=False,
                encoding="utf-8",
            ) as handle:
                handle.write(flattened)
                temp_path = Path(handle.name)
                render_source = temp_path

    command = [dot_binary]
    if dpi is not None:
        command.append(f"-Gdpi={dpi}")
    command.extend(["-Tpng", str(render_source), "-o", str(png_path)])
    subprocess.run(command, check=True)

    if temp_path is not None and temp_path.exists():
        temp_path.unlink()
    return True


def plot_feature_importance(rows: list[dict[str, Any]], output_path: Path = IMPORTANCE_PNG_PATH) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_rows = rows[:10]
    labels = [row["source_feature"] for row in reversed(top_rows)]
    values = [row["importance"] for row in reversed(top_rows)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color="#4f7cac")
    ax.set_title("Teacher Tree Aggregated Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
