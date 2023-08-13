from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import aiofiles
from app.preprocess_image import preprocess_image
import tensorflow as tf
import numpy as np


class SkinTypeResponse(BaseModel):
    skin_type: str


app = FastAPI()

pigmentation_model = tf.keras.models.load_model('models/pigmented_nonpigmented_model.keras')
oily_model = tf.keras.models.load_model('models/oily_dry_model.keras')
sensation_model = tf.keras.models.load_model('models/sensation_model.keras')
wrinkles_model = tf.keras.models.load_model('models/wrinkles_model.keras')


@app.post("/macro")
async def predict_macro(file: UploadFile):
    async with aiofiles.open('image.jpg', 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    image = preprocess_image('image.jpg')
    is_pigmented = pigmentation_model.predict(image)
    is_sensitive = sensation_model.predict(image)
    is_oily = oily_model.predict(image)
    is_wrinkled = wrinkles_model.predict(image)
    skin_type = 'O' if np.argmax(is_oily) else 'D'
    skin_type += 'S' if np.argmax(is_sensitive) else 'R'
    skin_type += 'P' if np.argmax(is_pigmented) else 'N'
    skin_type += 'W' if np.argmax(is_wrinkled) else 'T'
    response = SkinTypeResponse(
        skin_type=skin_type
    )
    return response
