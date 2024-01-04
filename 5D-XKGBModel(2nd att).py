import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# Load the data
data = np.load('combined_data_5D.npy')

# Separate features (X) and labels (y)
X = data[:, :4]
y = data[:, 4]

# Map labels to positive range
label_range = np.arange(len(np.unique(y)))
label_map = dict(zip(np.unique(y), label_range))
y = np.array([label_map[i] for i in y])

# Calculate the number of repetitions required for each class to have at least
# the average number of samples per class
class_counts = np.bincount(y)
avg_count = int(np.mean(class_counts))
repetitions = [int(np.ceil(avg_count / count)) for count in class_counts]

# Repeat the samples in each class by the required number of repetitions
X_repeated = np.empty((0, X.shape[1]))
y_repeated = np.empty(0, dtype=int)

for label, rep in enumerate(repetitions):
    X_class = X[y == label]
    y_class = y[y == label]
    X_repeated = np.concatenate((X_repeated, np.repeat(X_class, rep, axis=0)), axis=0)
    y_repeated = np.concatenate((y_repeated, np.repeat(y_class, rep, axis=0)), axis=0)

# Use StratifiedShuffleSplit to ensure even distribution of classes in the training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X_repeated, y_repeated):
    X_train, X_test = X_repeated[train_index], X_repeated[test_index]
    y_train, y_test = y_repeated[train_index], y_repeated[test_index]

# Create a dynamic mapping of class labels based on the actual values in y
class_map = {}
for i, label in enumerate(np.unique(y_repeated)):
    class_map[i] = label

# Transform y using the dynamic mapping
y_train_transformed = np.array([np.where(np.array(list(class_map.values())) == label)[0][0] for label in y_train])
y_test_transformed = np.array([np.where(np.array(list(class_map.values())) == label)[0][0] for label in y_test])

# Create the XGBoost classifier with a single set of hyperparameters
clf = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=1000, eval_metric='merror')

# Fit the model on the training data with verbose output
eval_set = [(X_train, y_train_transformed), (X_test, y_test_transformed)]
clf.fit(X_train, y_train_transformed, eval_set=eval_set, verbose=True)

# Predict the test data
y_pred = clf.predict(X_test)

# Map the predicted labels back to their original class labels
y_pred_orig = np.array([class_map[i] for i in y_pred])

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'Accuracy: {accuracy:.4f}')


