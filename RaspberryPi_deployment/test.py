
import joblib
model = joblib.load('rf_day1.pkl')
print('Number of features:', model.n_features_in_)
print('Feature names:')
for f in model.feature_names_in_:
    print(' -', f)
