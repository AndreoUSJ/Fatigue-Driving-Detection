import tensorflow as tf

model = tf.keras.models.load_model("model_2_simple_cnn.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: apply float16 quantisation for smaller APK
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("fatigue_model2.tflite", "wb") as f:
    f.write(tflite_model)
 
print("Saved fatigue_model2.tflite")