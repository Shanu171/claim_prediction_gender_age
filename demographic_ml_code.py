# final_claim_model_pipeline_with_importance.py
import pandas as pd
import numpy as np
import joblib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
import os

# --------------------------------------------------------
# 1️⃣ Load or Create Data
# --------------------------------------------------------
df = pd.read_csv("synthetic_claim_data.csv")

# Add some extra categorical columns
np.random.seed(42)
df["Scheme Category"] = np.random.choice(["Bronze", "Silver", "Gold", "Platinum", "Diamond"], size=len(df))
df["Member Status"] = np.random.choice(["Active", "Inactive", "Suspended"], size=len(df))
df["Registration Status"] = np.random.choice(["Registered", "Unregistered"], size=len(df))
df["Short Post Code"] = np.random.choice([f"PC_{i:02d}" for i in range(1, 72)], size=len(df))

TARGET = "Claim Amount"
X = df.drop(columns=[TARGET])
y = np.log1p(df[TARGET])  # log-transform to stabilize

# --------------------------------------------------------
# 2️⃣ Column Groups
# --------------------------------------------------------
ordinal_cols = ["Scheme Category"]
onehot_cols = ["Claimant Gender", "Member Status", "Registration Status"]
target_cols = ["Short Post Code"]
numeric_cols = ["Claimant Age"]

# --------------------------------------------------------
# 3️⃣ Hybrid Preprocessor
# --------------------------------------------------------
ordinal_pipe = Pipeline([
    ("encoder", OrdinalEncoder(categories=[["Bronze", "Silver", "Gold", "Platinum", "Diamond"]]))
])

onehot_pipe = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

target_pipe = Pipeline([
    ("encoder", TargetEncoder())
])

num_pipe = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("ordinal", ordinal_pipe, ordinal_cols),
    ("onehot", onehot_pipe, onehot_cols),
    ("target", target_pipe, target_cols),
    ("num", num_pipe, numeric_cols)
])

# --------------------------------------------------------
# 4️⃣ Train-Test Split
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------
# 5️⃣ Model + RandomizedSearchCV
# --------------------------------------------------------
model = LGBMRegressor(random_state=42)

param_grid = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [5, 8, 12, -1],
    "num_leaves": [31, 63, 127],
    "subsample": [0.7, 0.9, 1.0]
}

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

cv = KFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=10,
    cv=cv,
    scoring="neg_mean_absolute_error",
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("🔍 Training LightGBM model...")
search.fit(X_train, y_train)
best_model = search.best_estimator_
print("✅ Best Params:", search.best_params_)

# --------------------------------------------------------
# 6️⃣ Evaluation
# --------------------------------------------------------
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print("\n📊 Evaluation Results:")
print(f"MAE:  ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²:   {r2:.4f}")

# --------------------------------------------------------
# 7️⃣ Feature Importance - LightGBM Built-in
# --------------------------------------------------------
print("\n🎯 Computing Feature Importances...")

# Extract preprocessed feature names
ohe = best_model.named_steps["preprocessor"].named_transformers_["onehot"].named_steps["encoder"]
onehot_feature_names = ohe.get_feature_names_out(onehot_cols)

all_feature_names = (
    ordinal_cols +
    list(onehot_feature_names) +
    target_cols +
    numeric_cols
)

# Get model from pipeline
lgb_model = best_model.named_steps["model"]

importance_gain = lgb_model.booster_.feature_importance(importance_type='gain')
importance_split = lgb_model.booster_.feature_importance(importance_type='split')

feat_imp = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance (Gain)": importance_gain,
    "Importance (Split)": importance_split
}).sort_values(by="Importance (Gain)", ascending=False)

print("\n📈 Top 10 Important Features:")
print(feat_imp.head(10))

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp.head(15), x="Importance (Gain)", y="Feature", color="skyblue")
plt.title("Top Feature Importances (LightGBM Gain)")
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 8️⃣ Permutation Importance (Model-Agnostic)
# --------------------------------------------------------
print("\n🔁 Calculating Permutation Importance...")
perm_result = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]
top_perm_features = np.array(all_feature_names)[perm_sorted_idx[:10]]

plt.figure(figsize=(10,6))
plt.barh(top_perm_features[::-1], perm_result.importances_mean[perm_sorted_idx[:10]][::-1])
plt.title("Top 10 Features (Permutation Importance)")
plt.xlabel("Mean Importance Decrease")
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 9️⃣ SHAP Explainability (Local + Global)
# --------------------------------------------------------
print("\n⚡ Generating SHAP values...")
explainer = shap.Explainer(lgb_model)
X_test_pre = best_model.named_steps["preprocessor"].transform(X_test)
shap_values = explainer(X_test_pre)

# Global importance
shap.summary_plot(shap_values, features=X_test_pre, feature_names=all_feature_names, show=True)

# Local explanation for one example
print("\n💡 Example SHAP explanation for one prediction:")
shap.plots.waterfall(shap_values[0], max_display=10)

# --------------------------------------------------------
# 🔟 Save model
# --------------------------------------------------------
os.makedirs("models", exist_ok=True)
model_path = "models/final_claim_model_with_importance.pkl"
joblib.dump(best_model, model_path)
print(f"\n💾 Model saved successfully at: {model_path}")
