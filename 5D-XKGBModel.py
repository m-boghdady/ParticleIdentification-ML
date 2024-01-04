import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# Load the data
data = np.load('combined_data_5D.npy')

# Separate features (X) and labels (y)
X = data[:, :4]  
y = data[:, 4]   

# Map the actual class labels to the desired class labels
class_map = {111: 0, 112: 1, 113: 2, 114: 3, 115: 4, 116: 5, 121: 6, 122: 7, 123: 8, 201: 9, 202: 10}
y_transformed = np.array([class_map[label] for label in y])

# Use StratifiedShuffleSplit to ensure even distribution of classes in the training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, y_transformed):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_transformed[train_index], y_transformed[test_index]

# Create the XGBoost classifier with a single set of hyperparameters
clf = xgb.XGBClassifier(max_depth=9, learning_rate=0.1, n_estimators=100, eval_metric='merror')

# Fit the model on the training data with verbose output
eval_set = [(X_train, y_train), (X_test, y_test)]
clf.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Predict the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'Accuracy: {accuracy:.4f}')

