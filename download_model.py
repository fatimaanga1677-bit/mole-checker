import tensorflow as tf

# Load pre-trained MobileNetV2 model (ImageNet weights)
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224,224,3))

# Save the model locally for your Streamlit app
model.save("mobilenet_v2_224.keras")

print("✅ MobileNetV2 model downloaded and saved locally!")
