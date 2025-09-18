import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load preprocessed dataset
data = pd.read_csv("preprocessed.csv")

# Features and target
X = data.drop(columns=['Yield(tons)'])
y = data['Yield(tons)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
ln_clf = LinearRegression()
rf_clf = RandomForestRegressor(random_state=42)
gb_clf = GradientBoostingRegressor(random_state=42)


# ---------- Evaluation Function ----------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # ✅ RMSE fixed
    mae = mean_absolute_error(y_test, y_pred)

    return r2, mse, rmse, mae


# ---------- Base Model Evaluation ----------
ln_results = evaluate_model(ln_clf, X_train, X_test, y_train, y_test)
rf_results = evaluate_model(rf_clf, X_train, X_test, y_train, y_test)
gb_results = evaluate_model(gb_clf, X_train, X_test, y_train, y_test)

print("Linear Regression - R2: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}, MAE: {:.2f}".format(*ln_results))
print("Random Forest      - R2: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}, MAE: {:.2f}".format(*rf_results))
print("Gradient Boosting  - R2: {:.2f}, MSE: {:.2f}, RMSE: {:.2f}, MAE: {:.2f}".format(*gb_results))


# ---------- Hyperparameter Tuning ----------
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(
    estimator=rf_clf, param_grid=rf_param_grid,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
rf_grid_search.fit(X_train, y_train)
print("Best Random Forest Hyperparameters:", rf_grid_search.best_params_)

gb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
gb_grid_search = GridSearchCV(
    estimator=gb_clf, param_grid=gb_param_grid,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
gb_grid_search.fit(X_train, y_train)
print("Best Gradient Boosting Hyperparameters:", gb_grid_search.best_params_)


# ---------- Cross Validation ----------
rf_best_model = rf_grid_search.best_estimator_
gb_best_model = gb_grid_search.best_estimator_

rf_cv_scores = cross_val_score(rf_best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Random Forest Cross-Validation MSE: {-rf_cv_scores.mean():.2f} ± {rf_cv_scores.std():.2f}")

gb_cv_scores = cross_val_score(gb_best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Gradient Boosting Cross-Validation MSE: {-gb_cv_scores.mean():.2f} ± {gb_cv_scores.std():.2f}")


# ---------- Final Evaluation ----------
rf_best_model.fit(X_train, y_train)
gb_best_model.fit(X_train, y_train)

rf_pred = rf_best_model.predict(X_test)
gb_pred = gb_best_model.predict(X_test)

# Random Forest Metrics
rf_r2 = r2_score(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)

# Gradient Boosting Metrics
gb_r2 = r2_score(y_test, gb_pred)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_rmse = np.sqrt(gb_mse)
gb_mae = mean_absolute_error(y_test, gb_pred)

print(f"\nFinal Random Forest - R2: {rf_r2:.2f}, MSE: {rf_mse:.2f}, RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")
print(f"Final Gradient Boosting - R2: {gb_r2:.2f}, MSE: {gb_mse:.2f}, RMSE: {gb_rmse:.2f}, MAE: {gb_mae:.2f}")


# ---------- Save Models & Encoders ----------
crop_encoder = LabelEncoder()
irrigation_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
season_encoder = LabelEncoder()

with open('random_forest_agri_model.pkl', 'wb') as f:
    pickle.dump(rf_best_model, f)   # ✅ save tuned best model

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(crop_encoder, f)

with open('irrigation_encoder.pkl', 'wb') as f:
    pickle.dump(irrigation_encoder, f)

with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(soil_encoder, f)

with open('season_encoder.pkl', 'wb') as f:
    pickle.dump(season_encoder, f)

# StandardScaler for scaling numerical features
scaler = StandardScaler().fit(
    X_train[["Farm_Area(acres)", "Fertilizer_Used(tons)", "Pesticide_Used(kg)", "Water_Usage(cubic meters)"]]
)

with open('scaler_agri.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the list of column names
with open("columns_order_agri.pkl", "wb") as f:
    pickle.dump(list(X_train.columns), f)
