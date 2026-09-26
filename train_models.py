"""
Train and evaluate models that predict which sorting
algorithm suits an input.

The evaluation deliberately reports two different things,
because they answer different questions.

Evaluation 1
    Repeated stratified 80/20 hold-out, plus stratified
    k-fold cross-validation.

    "Can the model predict an input drawn from a
    distribution it has already seen?"

Evaluation 2
    Leave-one-distribution-out. The input type is used as
    the group, so every fold trains on six input types and
    tests on the seventh.

    "Can the model generalise to a distribution it has
    never seen during training?"

A note on interpreting the scores
---------------------------------
A perfect score is not automatically a good result, and a
lower score is not automatically a bad one. What matters is
whether the model is solving a real problem. The
"label purity" section below measures that directly: if one
single input type always produces the same winning
algorithm, the model only has to recognise the input type,
and the score says nothing about whether input
characteristics guide the choice.

There is no hyperparameter search here. Tuning against the
same split that reports the score would make the score
meaningless.
"""

import argparse

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "size",
    "runs",
    "longest_run",
    "duplicate_ratio",
    "value_range"
]

LABELS = [
    "performance_label",
    "energy_label",
    "hybrid_label"
]

HUMAN_NAMES = {
    "performance_label":
        "PERFORMANCE",
    "energy_label":
        "ENERGY",
    "hybrid_label":
        "HYBRID"
}

# Fixed in advance, not tuned against the reported score.

DECISION_TREE_PARAMS = {
    "max_depth": 8,
    "random_state": 42
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "random_state": 42
}

N_HOLDOUT_REPEATS = 10
N_CV_SPLITS = 5
N_CV_REPEATS = 3

TEST_SIZE = 0.20


def build_models():

    return {
        "Decision Tree": DecisionTreeClassifier(
            **DECISION_TREE_PARAMS
        ),
        "Random Forest": RandomForestClassifier(
            **RANDOM_FOREST_PARAMS
        )
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print(f"Path: {path}")
    print(f"Rows: {len(df)}")

    missing = [
        column
        for column in FEATURES + LABELS
        if column not in df.columns
    ]

    if missing:
        print()
        print(
            f"ERROR: missing columns: {missing}"
        )
        print(
            "Regenerate the dataset with "
            "experiment.py"
        )
        raise SystemExit(1)

    print()
    print("Input types:")

    for name, count in df[
        "input_type"
    ].value_counts().items():

        print(f"  {name:20s} {count}")

    print()
    print("Input sizes:")

    for size, count in df[
        "size"
    ].value_counts().sort_index().items():

        print(f"  {size:>20} {count}")

    return df


# ============================================================
# DIAGNOSTIC
#
# Is the problem still trivial?
# ============================================================

def report_label_purity(df):
    """
    How often does one input type produce the same
    winning algorithm?

    A purity near 100% means the label can be read off
    the input type alone. The model then only has to
    identify the input type, and a high accuracy
    carries no information about whether input
    characteristics guide the choice.
    """

    print()
    print("=" * 60)
    print("DIAGNOSTIC: IS THE PROBLEM STILL TRIVIAL?")
    print("=" * 60)

    for column in LABELS:

        print()
        print(HUMAN_NAMES[column])

        purities = []

        for input_type, group in df.groupby(
            "input_type"
        ):

            counts = group[
                column
            ].value_counts()

            majority = counts.iloc[0]

            purity = majority / len(group)

            purities.append(purity)

            summary = ", ".join(
                f"{name}={count}"
                for name, count in counts.items()
            )

            print(
                f"  {input_type:18s} "
                f"purity={purity:5.1%}  "
                f"{summary}"
            )

        mean_purity = np.mean(purities)

        print()
        print(
            f"  mean purity: {mean_purity:.1%}"
        )

        if mean_purity > 0.95:

            print(
                "  WARNING: the winning algorithm is "
                "almost"
            )

            print(
                "  determined by the input type. A model "
                "can score"
            )

            print(
                "  highly here by recognising the type "
                "alone."
            )

        elif mean_purity > 0.80:

            print(
                "  The input type is informative but "
                "does not"
            )

            print(
                "  fully determine the winner."
            )

        else:

            print(
                "  The winner varies within input types. "
                "The model"
            )

            print(
                "  has to use the features to succeed."
            )


def report_label_agreement(df):
    """
    The three labels are only worth training separately
    if they actually differ.
    """

    print()
    print("=" * 60)
    print("DIAGNOSTIC: DO THE THREE MODES DIFFER?")
    print("=" * 60)

    pairs = [
        ("performance_label", "energy_label"),
        ("performance_label", "hybrid_label"),
        ("energy_label", "hybrid_label")
    ]

    for left, right in pairs:

        agreement = (
            df[left] == df[right]
        ).mean()

        print()
        print(
            f"  {left} == {right}:  "
            f"{agreement:.1%}"
        )

        if agreement > 0.99:

            print(
                "    These two modes are the same "
                "experiment."
            )

        elif agreement > 0.90:

            print(
                "    These two modes are nearly "
                "identical."
            )


# ============================================================
# REPORTING HELPERS
# ============================================================

def rule(char="-"):

    print(char * 60)


def usable_splits(y, desired):
    """
    The largest number of stratified folds that the class
    counts actually support.

    Returns None when stratification is impossible because
    some class has a single member.
    """

    counts = y.value_counts()

    if len(counts) == 0:
        return None

    smallest = counts.min()

    if smallest < 2:
        return None

    return int(
        min(desired, smallest)
    )


def check_stratification(y, label_name):
    """
    Warn when a label is too rare to be learned.

    A winning algorithm that appears only once or twice
    cannot be predicted reliably, no matter which model is
    used. That is a property of the experiment, not a bug,
    but it has to be reported rather than crashed on.
    """

    counts = y.value_counts()

    rare = counts[
        counts < 2
    ]

    if len(rare) == 0:
        return True

    print()
    print(
        f"  WARNING: in {HUMAN_NAMES[label_name]}, "
        "these labels are too rare to stratify on:"
    )

    for name, count in rare.items():

        print(
            f"    {name:16s} appears {count} time(s)"
        )

    print(
        "    Stratification is disabled for this "
        "label and any"
    )

    print(
        "    score involving it should be read with "
        "care."
    )

    return False


def print_score_line(
    model_name,
    label_name,
    mean,
    std
):

    print(
        f"    {model_name:16s} "
        f"{mean:6.1%}"
    )

    if std is not None:

        print(
            f"    {'':16s} "
            f"+/- {std:.1%} std"
        )


# ============================================================
# EVALUATION 1
#
# Repeated stratified hold-out and cross-validation.
# ============================================================

def evaluate_holdout(
    X,
    y,
    label_name
):

    print()
    print(
        "  Repeated stratified "
        f"{int((1 - TEST_SIZE) * 100)}/"
        f"{int(TEST_SIZE * 100)} hold-out, "
        f"{N_HOLDOUT_REPEATS} repeats"
    )

    can_stratify = check_stratification(
        y,
        label_name
    )

    stratify = (
        y
        if can_stratify
        else None
    )

    for model_name, model in (
        build_models().items()
    ):

        scores = []

        for repeat in range(
            N_HOLDOUT_REPEATS
        ):

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=TEST_SIZE,
                    random_state=42 + repeat,
                    stratify=stratify
                )
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            scores.append(
                accuracy_score(
                    y_test,
                    predictions
                )
            )

        print_score_line(
            model_name,
            label_name,
            np.mean(scores),
            np.std(scores)
        )

    print()
    print(
        f"  Stratified {N_CV_SPLITS}-fold "
        f"cross-validation, "
        f"{N_CV_REPEATS} repeats"
    )

    n_splits = usable_splits(
        y,
        N_CV_SPLITS
    )

    if n_splits is None:

        print(
            "    Skipped: a class is too rare to "
            "stratify on."
        )

        return

    print(
        f"    Using {n_splits} folds, the "
        "largest the class"
    )

    print(
        "    counts support."
    )

    for model_name, model in (
        build_models().items()
    ):

        cross = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=N_CV_REPEATS,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cross,
            scoring="accuracy"
        )

        print_score_line(
            model_name,
            label_name,
            scores.mean(),
            scores.std()
        )


# ============================================================
# EVALUATION 2
#
# Leave-one-distribution-out.
# ============================================================

def evaluate_holdout_distribution(
    X,
    y,
    groups,
    label_name
):

    print()
    print(
        "  Leave-one-distribution-out"
    )

    print(
        "  (train on six input types, test on the seventh)"
    )

    splitter = LeaveOneGroupOut()

    for model_name, model in (
        build_models().items()
    ):

        print()
        print(f"    {model_name}")

        fold_scores = []

        for held_out, (
            train_index,
            test_index
        ) in enumerate(
            splitter.split(X, y, groups)
        ):

            held_out_name = groups.iloc[
                test_index
            ].iloc[0]

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            # A model can only ever predict a class it has
            # seen. If the held-out type always produces a
            # winning algorithm that never appears in
            # training, the fold has a ceiling below 100%
            # for a reason that has nothing to do with the
            # features.

            unseen = set(y_test) - set(y_train)

            if unseen:

                ceiling = 1.0 - (
                    len(
                        y_test[
                            y_test.isin(unseen)
                        ]
                    ) / len(y_test)
                )

                note = (
                    f"ceiling {ceiling:.0%}, "
                    f"unseen: "
                    f"{sorted(unseen)}"
                )

            else:
                note = "all classes seen"

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = accuracy_score(
                y_test,
                predictions
            )

            fold_scores.append(score)

            print(
                f"      held out {held_out_name:18s} "
                f"n={len(y_test):>4}  "
                f"accuracy={score:6.1%}  {note}"
            )

        print(
            f"      mean = {np.mean(fold_scores):.1%} "
            f"+/- {np.std(fold_scores):.1%}"
        )


# ============================================================
# FULL REPORT FOR ONE LABEL
# ============================================================

def report_label(
    df,
    label_name
):

    print()
    print("=" * 60)
    print(f"MODE: {HUMAN_NAMES[label_name]}")
    print("=" * 60)

    print()
    print("Label distribution")

    for name, count in df[
        label_name
    ].value_counts().items():

        share = count / len(df)

        print(
            f"  {name:16s} "
            f"{count:>5}  {share:6.1%}"
        )


    # --------------------------------------------------------
    # A model that always answers with the most common
    # label. Without this, an accuracy cannot be judged.
    # --------------------------------------------------------

    X = df[FEATURES]
    y = df[label_name]
    groups = df["input_type"]

    print()
    print(
        "  Baseline: always predict the most "
        "common label"
    )

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline_splits = usable_splits(
        y,
        N_CV_SPLITS
    )

    if baseline_splits is None:

        baseline_score = np.full(
            N_HOLDOUT_REPEATS,
            y.value_counts(normalize=True).max()
        )

    else:

        cross = RepeatedStratifiedKFold(
            n_splits=baseline_splits,
            n_repeats=N_CV_REPEATS,
            random_state=42
        )

        baseline_scores = cross_val_score(
            baseline,
            X,
            y,
            cv=cross,
            scoring="accuracy"
        )

        baseline_score = baseline_scores

    print(
        f"    {'Dummy':16s} "
        f"{baseline_score.mean():6.1%} "
        f"+/- {baseline_score.std():.1%} std"
    )


    # --------------------------------------------------------
    # Evaluation 1
    # --------------------------------------------------------

    print()
    rule()
    print(
        "EVALUATION 1: inputs from "
        "distributions already seen"
    )
    rule()

    evaluate_holdout(
        X,
        y,
        label_name
    )


    # --------------------------------------------------------
    # Evaluation 2
    # --------------------------------------------------------

    print()
    rule()
    print(
        "EVALUATION 2: inputs from a "
        "distribution never seen"
    )
    rule()

    evaluate_holdout_distribution(
        X,
        y,
        groups,
        label_name
    )


    # --------------------------------------------------------
    # Detail for the repeated hold-out
    # --------------------------------------------------------

    print()
    rule()
    print(
        "DETAIL: classification report on the "
        "final hold-out split"
    )
    rule()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=42,
            stratify=(
                y
                if usable_splits(y, 2)
                else None
            )
        )
    )

    print(
        f"  Training rows: {len(X_train)}"
    )

    print(
        f"  Testing rows:  {len(X_test)}"
    )

    for model_name, model in (
        build_models().items()
    ):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print()
        print(f"  {model_name}")
        print()

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        print(
            "  Confusion matrix "
            "(rows = actual, columns = predicted)"
        )

        print()

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=sorted(y.unique())
        )

        classes = sorted(y.unique())

        header = "".join(
            f"{name[:8]:>10}"
            for name in classes
        )

        print(f"{'':>10}{header}")

        for name, row in zip(
            classes,
            matrix
        ):

            cells = "".join(
                f"{value:>10}"
                for value in row
            )

            print(
                f"{name[:10]:>10}{cells}"
            )

        print()
        print(
            "  Feature importance"
        )

        importances = (
            model.feature_importances_
        )

        for feature, importance in sorted(
            zip(
                FEATURES,
                importances
            ),
            key=lambda pair: -pair[1]
        ):

            bar = "#" * int(
                round(importance * 40)
            )

            print(
                f"    {feature:18s} "
                f"{importance:6.4f}  {bar}"
            )


# ============================================================
# MAIN
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Predict the best sorting algorithm "
            "from input characteristics."
        )
    )

    parser.add_argument(
        "--ml-csv",
        default="ml_dataset.csv",
        help=(
            "Path to the ML dataset. "
            "Default: ml_dataset.csv"
        )
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    df = load_dataset(args.ml_csv)

    report_label_agreement(df)

    report_label_purity(df)

    for label_name in LABELS:

        report_label(
            df,
            label_name
        )

    print()
    print("=" * 60)
    print(
        "INTERPRETING THESE RESULTS"
    )
    print("=" * 60)

    print()
    print(
        "  A high score in Evaluation 1 and a much"
    )
    print(
        "  lower score in Evaluation 2 means the "
        "model has"
    )

    print(
        "  learned the input types it was trained on "
        "rather than"
    )

    print(
        "  the relationship between the features and "
        "the winner."
    )

    print()
    print(
        "  Compare every model against the Dummy "
        "baseline, not"
    )

    print(
        "  against 100%. A model that beats the "
        "baseline only"
    )

    print(
        "  slightly has only learned the class "
        "frequencies."
    )

    print()
    print(
        "  Check the label purity first. If purity is "
        "high, a"
    )

    print(
        "  high score mostly reflects recognising the "
        "input type."
    )

    print()
    print("=" * 60)


if __name__ == "__main__":

    main()
