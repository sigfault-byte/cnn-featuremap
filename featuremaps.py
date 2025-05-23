import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load and Preprocess the Image
img_path = tf.keras.utils.get_file('cat.jpg', 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg')
img = image.load_img(img_path, target_size=(224, 224))  # Resize for VGG16
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  # Normalize input for VGG16

# Run Image Through the Feature Map Model
feature_maps = feature_map_model.predict(img_array)

print(f"Feature Map Shape: {feature_maps.shape}")  # Should be (1, height, width, channels)


# Number of Feature Maps
num_filters = feature_maps.shape[-1]  # Number of feature maps (channels)

plt.figure(figsize=(20, 20))
for i in range(min(num_filters, 16)):  # Show first 16 feature maps
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')

plt.suptitle(f"Feature Maps from Layer: {layer_name}", fontsize=20)
plt.savefig("feature_maps.png")  # Save the plot instead of showing it
print("Feature maps saved as feature_maps.png")

