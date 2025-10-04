import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("cat_vs_dog.keras")  # or .h5

# Test with a sample image
img = Image.open("dhruv2.jpg").resize((128,128))
img_array = np.array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print(pred)
