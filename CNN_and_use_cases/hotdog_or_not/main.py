import os
import shutil

import kagglehub
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")
print("Dataset path:", path)

all_data_dir = os.path.join(path, "all_data")
os.makedirs(all_data_dir, exist_ok=True)

for category in ["hot_dog", "not_hot_dog"]:
    src_train = os.path.join(path, "train", category)
    src_test = os.path.join(path, "test", category)
    dst = os.path.join(all_data_dir, category)
    os.makedirs(dst, exist_ok=True)

    for src_folder in [src_train, src_test]:
        if os.path.exists(src_folder):
            for fname in os.listdir(src_folder):
                src_path = os.path.join(src_folder, fname)
                dst_path = os.path.join(dst, fname)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)

img_size = (224, 224)
batch_size = 16

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    directory=all_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    seed=42
)

val_generator = datagen.flow_from_directory(
    directory=all_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    seed=42,
    shuffle=False
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
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
images_rgb = (images[:5] + 1.0) / 2.0
preds = model.predict(images[:5])

plt.figure(figsize=(12, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images_rgb[i])
    pred_label = class_names[int(preds[i][0] > 0.5)]
    true_label = class_names[int(labels[i])]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
