import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import layers, models
from tensorflow.math import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import cv2
import warnings
warnings.filterwarnings('ignore')


train_path = "train"
train_images = os.listdir(train_path)
train_label = pd.read_csv("labels.csv")
data = train_label.merge(pd.DataFrame({"id": [os.path.splitext(file)[0] for file in train_images]}), on="id")
print(data.head())

label_encoder = LabelEncoder()
data['encoded_labels'] = label_encoder.fit_transform(data['breed'])
output_folder_path = "processed_images"
os.makedirs(output_folder_path, exist_ok=True)
for image_name in os.listdir(train_path):
    image_path = os.path.join(train_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image_normalized = image / 255.0
    output_image_path = os.path.join(output_folder_path, f"processed_{image_name}")
    cv2.imwrite(output_image_path, image_normalized)

X = image_normalized
y = data['encoded_labels']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_train_normalized = X_train / 255.0
X_val_normalized = X_val / 255.0
X_test_normalized = X_test / 255.0

nrow = 5
ncol = 4
fig1 = plt.figure(figsize=(15, 15))
fig1.suptitle('After Resizing', size=32)
for i in range(20):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(image_normalized[i])
    plt.axis('Off')
    plt.grid(False)
plt.show()

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])
augmented_images = data_augmentation(image_normalized)

fig2 = plt.figure(figsize=(15, 15))
fig2.suptitle('After Augmentation', size=32)
for i in range(20):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(augmented_images[i])
    plt.axis('Off')
    plt.grid(False)
plt.show()

class_values = data["category"] - 1
class_values.value_counts()

X_train, X_test, y_train, y_test = train_test_split(image_normalized, class_values,random_state=30)
print(X_train.shape, y_train.shape)

model = models.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(5, activation='softmax')
])


checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[checkpoint])

best_model = models.load_model('best_model.h5')

test_loss, best_test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print("\nBest Test Accuracy:", best_test_acc)

best_predictions = best_model.predict(X_test)
best_predicted_classes = np.argmax(best_predictions, axis=1)

best_cm = confusion_matrix(y_test, best_predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='g', cmap='Blues', xticklabels=data.values(), yticklabels=data.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()