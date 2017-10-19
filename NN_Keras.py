# Proper Explanation and of each line and information about each dependency is in Description file.
# Please read this code along with Description file. :)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Function that creates our baseline Neural Network model

def create_baseline():
    # Create Model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load dataset
dataframe = pd.read_csv("sonar.csv", header=None)
dataset = dataframe.values

# Split into X and Y
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# Mapping or Encoding values 'M' and 'R' to 0 and 1 using Label Encoder
le = LabelEncoder()
le.fit(Y)
enc_Y = le.transform(Y)

# Evaluate Model with Standardized Dataset
np.random.seed(seed)
est = []
est.append(('standardize', StandardScaler()))
est.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))

# Pipeling both standardization and model into one
pipeline = Pipeline(est)

# Generating cross validation splitting generator
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# calculating accuracy
results = cross_val_score(pipeline, X, enc_Y, cv=kfold)
print("Standardized Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
# Accuracy 84.09%
