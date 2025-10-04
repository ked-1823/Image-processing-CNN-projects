import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model=tf.keras.models.load_model('cat_vs_dog.keras')

#title for the app
st.title('cat vs dog classifier')
#uploading file 
st.write('upload the image of cat and dog and the model will predict')
upload_file=st.file_uploader('choose an image:',type=['jpg','png','jpeg'])

#reshaping the uploaded file according to our model requirement

if upload_file is not None:
	# Open the image
	img=Image.open(upload_file)
	#show the image
	st.image(img,caption='uploaded image',use_column_width=True)

	#resizing the image acc to model

	img=img.resize((128,128))
	img_array=np.array(img)/255.0  #normalizing the image for similer pixel numer range
	img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

	#prediction

	pred=model.predict(img_array)
	class_names=['cat','dog']
	pred_class=class_names[np.argmax(pred)]
	confidence=np.max(pred)*100
	st.write(f'prediction: {pred_class}')
	st.write(f'confidence :{confidence:.2f}%')