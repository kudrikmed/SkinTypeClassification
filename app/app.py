from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import aiofiles
from app.preprocess_image import preprocess_image
import numpy as np
import json
import os
import tensorflow as tf


class SkinTypeResponse(BaseModel):
    skin_type: str
    short_info: str


app = FastAPI()

pm_path = os.path.normpath('models/pigmented_nonpigmented_model.keras')
om_path = os.path.normpath('models/oily_dry_model.keras')
sm_path = os.path.normpath('models/sensation_model.keras')
wm_path = os.path.normpath('models/wrinkles_model.keras')

# load models
pigmentation_model = tf.keras.models.load_model(pm_path)
oily_model = tf.keras.models.load_model(om_path)
sensation_model = tf.keras.models.load_model(sm_path)
wrinkles_model = tf.keras.models.load_model(wm_path)


# load info
def get_text_info():
    f = open('text_info.json', encoding="utf-8")
    data = json.load(f)
    f.close()
    return data


text_info = get_text_info()


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
        skin_type=skin_type,
        short_info=text_info[skin_type]
    )
    return response
