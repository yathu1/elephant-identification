import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

# Define paths
train_dir = 'split_data/train'
val_dir = 'split_data/val'
test_dir = 'split_data/test'

# Parameters
img_height, img_width = 128, 128
batch_size = 32
seed = 42
num_folds = 5

# Create ImageDataGenerator instances
datagen = ImageDataGenerator(rescale=1./255)

# Load data
data_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed,
    shuffle=True
)

# K-Fold Cross Validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
fold_no = 1
for train_index, val_index in kf.split(data_generator):
    print(f'Training fold {fold_no}...')
    
    # Create train and validation generators
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed,
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed,
        subset='validation'
    )
    
    # Base model (InceptionV3)
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights=None, pooling='avg', input_shape=(128, 128, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())
    output = tf.keras.layers.Dense(len(class_names), activation='softmax')(model.output)
    new_model = tf.keras.Model(model.input, output)

    # Define early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        verbose=1,
        patience=2,
        mode='max',
        restore_best_weights=True
    )

    # Compile the model
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, reduction="auto", name="categorical_crossentropy"), 
                      metrics=['accuracy'])

    # Train the model
    history = new_model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[es_callback]
    )

    # Create test generator
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=seed,
        shuffle=False
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = new_model.evaluate(test_generator)
    print(f'Fold {fold_no} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    fold_no += 1

# Save the final model
new_model.save('InceptionV3_elephant_model.h5')