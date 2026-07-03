# Improved prediction.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    r2_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

def prediction(data):

    data = data.dropna().copy()

    st.header("🚀 Data Prediction")

    target = st.selectbox("Select Target Column", data.columns, width=300)

    features = st.multiselect(
        "Select Feature Columns",
        [c for c in data.columns if c != target],
        width=300,
    )

    if not features:
        return

    X = data[features].copy()
    y = data[target].copy()

    encoders = {}

    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    if y.dtype == "object":
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y.astype(str))
        st.session_state["target_encoder"] = y_encoder

    st.session_state["encoders"] = encoders

    problem_type = (
        "Classification"
        if data[target].dtype == "object" or data[target].nunique() < 10
        else "Regression"
    )

    st.info(f"Detected Problem Type: {problem_type}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data))
    c2.metric("Features", len(features))
    c3.metric("Target", target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    c1, c2 = st.columns(2)
    c1.metric("Training Samples", len(X_train))
    c2.metric("Testing Samples", len(X_test))

    if problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
        }

    if st.button("🚀 Train Models"):

        scores = {}
        trained = {}
        metrics = {}

        for name, model in models.items():

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            trained[name] = model

            if problem_type == "Regression":
                r2 = r2_score(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))

                scores[name] = r2
                metrics[name] = (r2, mae, rmse)

            else:
                acc = accuracy_score(y_test, pred)
                scores[name] = acc

        best_name = max(scores, key=scores.get)

        st.session_state["trained"] = True
        st.session_state["model"] = trained[best_name]
        st.session_state["best_name"] = best_name
        st.session_state["scores"] = scores
        st.session_state["problem"] = problem_type
        st.session_state["features"] = features
        st.session_state["cat_cols"] = list(encoders.keys())
        st.session_state["metrics"] = metrics

    if st.session_state.get("trained", False):

        problem_type = st.session_state["problem"]
        scores = st.session_state["scores"]

        metric_name = "R² Score" if problem_type == "Regression" else "Accuracy (%)"

        values = (
            list(scores.values())
            if problem_type == "Regression"
            else [v * 100 for v in scores.values()]
        )

        score_df = pd.DataFrame(
            {
                "Model": list(scores.keys()),
                metric_name: values,
            }
        )

        st.subheader("📊 Model Performance")
        st.dataframe(score_df, use_container_width=True)
        st.bar_chart(score_df.set_index("Model"))

        best_model = st.session_state["best_name"]

        if problem_type == "Regression":
            r2, mae, rmse = st.session_state["metrics"][best_model]

            st.success(f"🏆 Best Model: {best_model}")

            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{r2:.3f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("RMSE", f"{rmse:.2f}")

        else:
            st.success(
                f"🏆 Best Model: {best_model}\n\nAccuracy: {scores[best_model]*100:.2f}%"
            )

        st.divider()
        st.subheader("🔮 Prediction")

        model = st.session_state["model"]

        inputs = []

        for col in st.session_state["features"]:

            if col in st.session_state["cat_cols"]:

                le = st.session_state["encoders"][col]

                value = st.selectbox(col, list(le.classes_))
                inputs.append(le.transform([value])[0])

            else:

                inputs.append(st.number_input(col))

        if st.button("Predict"):

            result = model.predict([inputs])[0]

            if problem_type == "Classification" and "target_encoder" in st.session_state:
                result = st.session_state["target_encoder"].inverse_transform([int(result)])[0]

            st.subheader("🎯 Prediction Result")

            st.metric(target, result if problem_type=="Classification" else f"{float(result):.2f}")

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba([inputs]).max()*100
                st.info(f"Confidence : {confidence:.2f}%")

            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame(
                    {
                        "Feature": st.session_state["features"],
                        "Importance": model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)

                st.subheader("⭐ Feature Importance")
                st.bar_chart(imp.set_index("Feature"))
