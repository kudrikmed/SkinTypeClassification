from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import aiofiles
import os
import tensorflow as tf
import json
from app.preprocess_image import preprocess_image

# Create a FastAPI application
app = FastAPI()


# Define response model
class SkinTypeResponse(BaseModel):
    """
    Response model for skin type prediction.
    """
    skin_type: str
    short_info: str


# Define model paths
pm_path = os.path.normpath('models/pigmented_nonpigmented_model.h5')
om_path = os.path.normpath('models/oily_dry_model.h5')
sm_path = os.path.normpath('models/sensation_model.h5')
wm_path = os.path.normpath('models/wrinkles_model.h5')

# Load models
try:
    pigmentation_model = tf.keras.models.load_model(pm_path)
    oily_model = tf.keras.models.load_model(om_path)
    sensation_model = tf.keras.models.load_model(sm_path)
    wrinkles_model = tf.keras.models.load_model(wm_path)
except Exception as e:
    raise Exception(f"Error loading models: {str(e)}")


# Load text information
def get_text_info():
    """
    Load text information for skin types from a JSON file.
    """
    try:
        with open('text_info.json', encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Text info file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading text info: {str(e)}")


text_info = get_text_info()


# Endpoint for skin type prediction
@app.post("/macro")
async def predict_macro(file: UploadFile):
    """
    Predicts the skin type based on an uploaded image file.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        SkinTypeResponse: Predicted skin type and short information.
    """
    try:
        # Check if the uploaded file is a JPG or PNG image
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return HTTPException(status_code=400, detail="Only JPG or PNG files are allowed.")
        # Check if the uploaded file is less than 10 MB
        file_size = file.file.tell()
        if file_size > 10 * 1024 * 1024:
            return HTTPException(status_code=400, detail="File is too large. Upload a file less than 10 MB.")
        # Save the uploaded image
        async with aiofiles.open('image.jpg', 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Preprocess the image
        image = preprocess_image('image.jpg')

        # Make predictions using loaded models
        is_pigmented = pigmentation_model.predict(image)
        is_sensitive = sensation_model.predict(image)
        is_oily = oily_model.predict(image)
        is_wrinkled = wrinkles_model.predict(image)

        # Determine the skin type based on predictions
        skin_type = 'O' if is_oily[0][0] > 0.5 else 'D'
        skin_type += 'S' if is_sensitive[0][0] > 0.5 else 'R'
        skin_type += 'P' if is_pigmented[0][0] > 0.5 else 'N'
        skin_type += 'W' if is_wrinkled[0][0] > 0.5 else 'T'

        # Create a response object
        response = SkinTypeResponse(
            skin_type=skin_type,
            short_info=text_info[skin_type]
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
