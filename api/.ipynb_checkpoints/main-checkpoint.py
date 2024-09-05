from fastapi import FastAPI , File , UploadFile
import  uvicorn
import numpy as np 
from io import BytesIO
from  PIL import Image 
import tensorflow as tf 


app = FastAPI()
MODEL = tf.keras.models.load_model('./model.h5')
CLASS_NAME = ['Early Blight','Light Blight','Healthy']
import tensorflow as tf

# Load the model without compiling it
MODEL = tf.keras.models.load_model('./model.h5', compile=False)

# Recompile the model with a valid loss function and reduction argument
MODEL.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='sum_over_batch_size'))



# @app.get('/ping')
# async def ping():
#     return 'Hello'


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image 

# @app.post('/predict')
# async def predict(
#     file : UploadFile = File(...)
# ): 
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image,0)
#     prediction = MODEL.predict(img_batch)
#     print(prediction)

#     return  
    

# if __name__ == '__main__':
#     uvicorn.run(app, host='localhost',port=8080)