"""Train TabNet regressor and save submission predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetRegressor


def main() -> None:
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X = train_df.drop("exam_score", axis=1)
    if "id" in X.columns:
        X = X.drop("id", axis=1)
    y = train_df["exam_score"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col].astype(str))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = TabNetRegressor(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        seed=42,
    )

    model.fit(
        X_train.values.astype(np.float32),
        y_train.values.astype(np.float32).reshape(-1, 1),
        eval_set=[
            (
                X_val.values.astype(np.float32),
                y_val.values.astype(np.float32).reshape(-1, 1),
            )
        ],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        verbose=1
    )

    X_test = test_df.copy()
    if "exam_score" in X_test.columns:
        X_test = X_test.drop("exam_score", axis=1)
    if "id" in X_test.columns:
        X_test = X_test.drop("id", axis=1)

    test_predictions = model.predict(X_test.values.astype(np.float32)).reshape(-1)

    sample_submission = pd.read_csv("data/sample_submission.csv")
    submission = sample_submission.copy()
    submission["exam_score"] = test_predictions
    submission.to_csv("data/submission_tabnet.csv", index=False)

    print("TabNet submission created successfully!")
    print(submission.head(10))


if __name__ == "__main__":
    main()