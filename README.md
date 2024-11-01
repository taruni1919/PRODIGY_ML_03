import os

import numpy as np

import cv2

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Set paths

data_dir = 'path_to_data/train' # Change to your dataset path

images = []

labels = []

for filename in os.listdir(data_dir): img = cv2.imread(os.path.join(data_dir, filename)) img = cv2.resize(img, (64, 64)) # Resize to 64x64 pixels images.append(img.flatten()) # Flatten the image labels.append(0 if 'cat' in filename else 1) # Label 0 for cats, 1 for dogs

images = np.array(images) labels = np.array(labels)

# Step 2: Split the Data

4. Split into Training and Test Sets:

python

X_train, X_test, y_train, y_test = train_test_split(images, labels,
test_size=0.2, random_state=42)

#Step 3: Feature Scaling

5. Scale the Features:

python

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X test)

#Step 4: Train the SVM Model

6. Train the SVM:

python

from sklearn.svm import SVC

svm_model = SVC(kernel='linear') # You can also try 'rbf' or other kernels

svm_model.fit(X_train, y_train)

#Step 5: Evaluate the Model

7. Make Predictions:
python

from sklearn.metrics import classification_report, confusion_matrix

y_pred = svm_model.predict(X_test)

# Evaluate the model

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Step 6: Visualize Results

8. Visualize Some Predictions:

python

import matplotlib.pyplot as plt

def plot_images (images, labels, predictions=None):

plt.figure(figsize=(10, 10))

for i in range(9):

plt.subplot(3, 3, i + 1)

plt.imshow(images[i].reshape(64, 64, 3)) # Reshape to original size

plt.title(f"True: {'Dog' if labels[i] else 'Cat'}\nPred: {'Dog' if predictions[i] else 'Cat'}" if predictions is not None else "")

plt.axis('off')

plt.show()
plot_images(X_test, y_test, y_pred)

#Step 7: Hyperparameter Tuning (Optional)

9. Tuning the Model:

Consider using techniques like GridSearchCV to find optimal hyperparameters.

python

from sklearn.model_selection import GridSearchCV

param_grid = {

'C': [0.1, 1, 10],

'gamma': [0.001, 0.01, 0.1],

'kernel': ['linear', 'rbf']

}

grid = GridSearchCV (SVC(), param_grid, refit=True, verbose=2)

grid.fit(X_train, y_train)
