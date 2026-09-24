from __future__ import annotations

import json
from itertools import product

from sklearn.tree import DecisionTreeRegressor

from dt_modelsv4_utils import (
    BASELINE_DOT_PATH,
    BASELINE_PNG_PATH,
    BRANCH_FIGURES_DIR,
    FIGURES_DIR,
    IMPORTANCE_PNG_PATH,
    JSON_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RESULTS_DIR,
    TEACHER_DOT_PATH,
    TEACHER_PNG_PATH,
    TEACHER_TEXT_PATH,
    aggregate_feature_importances,
    build_interesting_teacher_branches,
    build_preprocessor,
    collect_split_feature_usage,
    export_tree_dot,
    export_tree_text_report,
    extract_leaf_rules,
    fit_naive_baseline,
    fit_naive_baseline_pipeline,
    load_complete_case_dataset,
    metric,
    plot_feature_importance,
    render_png_from_dot,
    root_split_summary,
    select_teacher,
    split_dataset,
)


FEATURE_EXCLUSION_RATIONALE = [
    {
        "feature": "Per_Capita_GDP",
        "reason": (
            "Removed from the final explanatory tree because it acts mainly as a macroeconomic regime proxy. "
            "Although predictive, it tends to encode broad time effects rather than transaction-level housing structure."
        ),
    },
    {
        "feature": "floor_area_sqm",
        "reason": (
            "Removed in this alternative specification to surface secondary structural, lease, and financing effects "
            "that may otherwise be dominated by the direct size signal."
        ),
    },
    {
        "feature": "region_URA",
        "reason": (
            "Removed in this alternative specification to avoid overlapping broad geography features and to let "
            "the tree rely on the remaining location indicators more parsimoniously."
        ),
    },
]

LEGACY_OUTPUTS = [
    RESULTS_DIR / "02_step2_teacher_feature_importance.png",
    RESULTS_DIR / "02_step2_teacher_tree.dot",
    RESULTS_DIR / "02_step2_teacher_tree.png",
    RESULTS_DIR / "03_step3_pruned_interpretable_tree.dot",
    RESULTS_DIR / "03_step3_pruned_interpretable_tree.png",
    RESULTS_DIR / "04_step4_final_evaluation_and_decision.ipynb",
    RESULTS_DIR / "04_step4_final_evaluation_and_decision.json",
    RESULTS_DIR / "05_sensitivity_without_floor_area.ipynb",
    FIGURES_DIR / "05_sensitivity_without_floor_area_tree.dot",
    FIGURES_DIR / "05_sensitivity_without_floor_area_tree.png",
    FIGURES_DIR / "05_sensitivity_without_floor_area_feature_importance.png",
]


def run_analysis() -> dict:
    df, data_info = load_complete_case_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    baseline = fit_naive_baseline(X_train, y_train, X_val, y_val, X_test, y_test)

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    teacher_rows: list[dict] = []
    for max_depth, min_samples_leaf, min_samples_split in product(
        [5, 6],
        [1000, 2000],
        [200, 500],
    ):
        model = DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        )
        model.fit(X_train_enc, y_train)
        teacher_rows.append(
            {
                "config": {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "min_samples_split": min_samples_split,
                    "ccp_alpha": 0.0,
                },
                "depth": int(model.get_depth()),
                "leaf_count": int(model.get_n_leaves()),
                "train": metric(y_train, model.predict(X_train_enc)),
                "validation": metric(y_val, model.predict(X_val_enc)),
                "test": metric(y_test, model.predict(X_test_enc)),
                "feature_importances": aggregate_feature_importances(feature_names, model.feature_importances_),
                "root_summary": root_split_summary(model, feature_names),
                "split_feature_usage": collect_split_feature_usage(model, feature_names),
                "model": model,
            }
        )

    teacher_rows.sort(key=lambda row: row["validation"]["rmse"])
    selected_teacher = select_teacher(teacher_rows)

    teacher_leaf_rules = extract_leaf_rules(selected_teacher["model"], feature_names)
    teacher_leaf_rules = sorted(
        teacher_leaf_rules,
        key=lambda row: (-row["samples"], -row["prediction"]),
    )
    interesting_branches = build_interesting_teacher_branches(selected_teacher["model"], feature_names)
    return {
        "data_prep": data_info,
        "features": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": [
                "region",
                "mature_estate",
                "flat_type",
                "flat_model",
            ],
            "macro_time_features": ["sora_monthly_avg"],
            "excluded_features": FEATURE_EXCLUSION_RATIONALE,
        },
        "step1_baseline": baseline,
        "step2_teacher_grid": [
            {
                key: value
                for key, value in row.items()
                if key != "model"
            }
            for row in teacher_rows
        ],
        "step2_selected_teacher": {
            key: value
            for key, value in selected_teacher.items()
            if key != "model"
        },
        "step3_teacher_interpretation": {
            "final_model_role": "teacher_tree",
            "depth": selected_teacher["depth"],
            "leaf_count": selected_teacher["leaf_count"],
            "root_summary": selected_teacher["root_summary"],
            "split_feature_usage": selected_teacher["split_feature_usage"],
            "leaf_rules": teacher_leaf_rules,
            "top_leaf_rules_by_samples": teacher_leaf_rules[:12],
            "interesting_branches": interesting_branches,
        },
        "artifacts": {
            "figures_dir": str(FIGURES_DIR),
            "branch_figures_dir": str(BRANCH_FIGURES_DIR),
            "baseline_dot_path": str(BASELINE_DOT_PATH),
            "baseline_png_path": str(BASELINE_PNG_PATH),
            "teacher_dot_path": str(TEACHER_DOT_PATH),
            "teacher_png_path": str(TEACHER_PNG_PATH),
            "teacher_text_path": str(TEACHER_TEXT_PATH),
            "teacher_importance_png_path": str(IMPORTANCE_PNG_PATH),
        },
    }


def write_outputs(summary: dict) -> dict[str, bool]:
    JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    df, _ = load_complete_case_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    baseline_pipeline = fit_naive_baseline_pipeline(X_train, y_train)
    baseline_feature_names = list(baseline_pipeline.named_steps["preprocessor"].get_feature_names_out())
    baseline_tree = baseline_pipeline.named_steps["model"]

    baseline_dot_source = export_tree_dot(
        baseline_tree,
        baseline_feature_names,
        title="Naive Baseline Tree",
        presentation="original",
    )
    rendered_baseline_tree = render_png_from_dot(
        baseline_dot_source,
        dot_path=BASELINE_DOT_PATH,
        png_path=BASELINE_PNG_PATH,
    )

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    feature_names = list(preprocessor.get_feature_names_out())

    teacher_config = summary["step2_selected_teacher"]["config"]
    teacher_model = DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        max_depth=teacher_config["max_depth"],
        min_samples_leaf=teacher_config["min_samples_leaf"],
        min_samples_split=teacher_config["min_samples_split"],
        ccp_alpha=teacher_config["ccp_alpha"],
    )
    teacher_model.fit(X_train_enc, y_train)

    teacher_dot_source = export_tree_dot(teacher_model, feature_names, presentation="original")
    rendered_teacher_tree = render_png_from_dot(
        teacher_dot_source,
        dot_path=TEACHER_DOT_PATH,
        png_path=TEACHER_PNG_PATH,
    )
    TEACHER_TEXT_PATH.write_text(export_tree_text_report(teacher_model, feature_names), encoding="utf-8")

    teacher_importance_rows = summary["step2_selected_teacher"]["feature_importances"]
    plot_feature_importance(teacher_importance_rows, output_path=IMPORTANCE_PNG_PATH)

    for old_branch_file in BRANCH_FIGURES_DIR.glob("*"):
        if old_branch_file.is_file():
            old_branch_file.unlink()

    for branch in summary["step3_teacher_interpretation"]["interesting_branches"]:
        branch_dot = BRANCH_FIGURES_DIR / f"{branch['name']}.dot"
        branch_png = BRANCH_FIGURES_DIR / f"{branch['name']}.png"
        branch["dot_path"] = str(branch_dot)
        branch["png_path"] = str(branch_png)
        branch_dot_source = export_tree_dot(
            teacher_model,
            feature_names,
            root_node=int(branch["node_id"]),
            title=branch["title"],
            presentation="original",
        )
        render_png_from_dot(
            branch_dot_source,
            dot_path=branch_dot,
            png_path=branch_png,
            unflatten_args=["-l", "4", "-c", "5"] if branch["name"] == "non_three_room_branch" else None,
            dpi=220 if branch["name"] == "non_three_room_branch" else None,
        )

    JSON_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for path in LEGACY_OUTPUTS:
        if path.exists():
            path.unlink()

    return {
        "rendered_baseline_tree_png": rendered_baseline_tree,
        "rendered_teacher_tree_png": rendered_teacher_tree,
    }


def main() -> None:
    summary = run_analysis()
    render_info = write_outputs(summary)

    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {BASELINE_DOT_PATH}")
    if render_info["rendered_baseline_tree_png"]:
        print(f"Wrote {BASELINE_PNG_PATH}")
    else:
        print("Graphviz 'dot' not found; baseline PNG was not rendered.")
    print(f"Wrote {TEACHER_DOT_PATH}")
    if render_info["rendered_teacher_tree_png"]:
        print(f"Wrote {TEACHER_PNG_PATH}")
    else:
        print("Graphviz 'dot' not found; teacher PNG was not rendered.")
    print(f"Wrote {TEACHER_TEXT_PATH}")
    print(f"Wrote {IMPORTANCE_PNG_PATH}")
    print("Selected teacher config:", summary["step2_selected_teacher"]["config"])


if __name__ == "__main__":
    main()
