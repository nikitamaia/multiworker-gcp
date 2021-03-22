import tensorflow as tf

def create_model(number_of_classes):
  base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1016, activation='relu')(x)
  predictions = tf.keras.layers.Dense(number_of_classes, activation='softmax')(x)
  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(0.0001),
      metrics=['accuracy'])

  return model
