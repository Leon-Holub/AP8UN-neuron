import os
import shutil
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
data_path = os.path.join(path)
print("Dataset path:", data_path)

readme_path = os.path.join(data_path, "README_rpc-cv-images.txt")
if os.path.isfile(readme_path):
    os.remove(readme_path)

invalid_folder = os.path.join(data_path, "rps-cv-images")
if os.path.isdir(invalid_folder):
    shutil.rmtree(invalid_folder)

img_size = (100, 100)
batch_size = 16

class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
num_classes = len(class_folders)
print("Třídy:", class_folders)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=42
)

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,
    callbacks=[early_stop]
)

plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

class_names = list(train_generator.class_indices.keys())
images, labels = next(val_generator)
preds = model.predict(images[:5])

plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i])
    pred_label = class_names[np.argmax(preds[i])]
    true_label = class_names[np.argmax(labels[i])]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()