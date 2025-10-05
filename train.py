import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
# ============================
# 1. Data Preprocessing
# ============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = "images/train"
test_dir = "images/test"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode='binary')
test_gen = test_datagen.flow_from_directory(test_dir,
                                            target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary')

# ============================
# 2. Load Pre-trained ResNet50
# ============================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base model

#base_model.trainable = True


# Build Model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ...existing code...
# 3. Train Model

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10,
    callbacks=[early_stop]
)

#history = model.fit(train_gen, validation_data=test_gen, epochs=10)

# Save the trained model
model.save("model/brain_tumor_model.h5")
print("Model saved as brain_tumor_model.h5")
# ...existing code...

 