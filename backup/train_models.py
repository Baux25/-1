import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ml_dataset.csv")

print("Dataset:")
print(df)

print()
print("Number of samples:", len(df))


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "size",
    "runs",
    "longest_run",
    "duplicate_ratio",
    "value_range"
]

X = df[FEATURES]


# ============================================================
# TRAIN ONE MODE
# ============================================================

def train_mode(label_column, mode_name):

    print()
    print("=" * 60)
    print(mode_name)
    print("=" * 60)

    y = df[label_column]


    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


    # ========================================================
    # DECISION TREE
    # ========================================================

    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    decision_tree.fit(
        X_train,
        y_train
    )


    dt_predictions = decision_tree.predict(
        X_test
    )


    dt_accuracy = accuracy_score(
        y_test,
        dt_predictions
    )


    print()
    print("Decision Tree Accuracy:")
    print(
        f"{dt_accuracy:.4f}"
    )


    print()
    print("Decision Tree Report:")
    print(
        classification_report(
            y_test,
            dt_predictions,
            zero_division=0
        )
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    random_forest.fit(
        X_train,
        y_train
    )


    rf_predictions = random_forest.predict(
        X_test
    )


    rf_accuracy = accuracy_score(
        y_test,
        rf_predictions
    )


    print()
    print("Random Forest Accuracy:")
    print(
        f"{rf_accuracy:.4f}"
    )


    print()
    print("Random Forest Report:")
    print(
        classification_report(
            y_test,
            rf_predictions,
            zero_division=0
        )
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print()
    print("Decision Tree Feature Importance:")

    for feature, importance in zip(
        FEATURES,
        decision_tree.feature_importances_
    ):

        print(
            f"{feature:20s} "
            f"{importance:.4f}"
        )


    print()
    print("Random Forest Feature Importance:")

    for feature, importance in zip(
        FEATURES,
        random_forest.feature_importances_
    ):

        print(
            f"{feature:20s} "
            f"{importance:.4f}"
        )


    return decision_tree, random_forest


# ============================================================
# TRAIN ALL THREE MODES
# ============================================================

dt_performance, rf_performance = train_mode(
    "performance_label",
    "PERFORMANCE MODE"
)


dt_energy, rf_energy = train_mode(
    "energy_label",
    "ENERGY MODE"
)


dt_hybrid, rf_hybrid = train_mode(
    "hybrid_label",
    "HYBRID MODE"
)


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)