from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import aiofiles
from src.models import predict_model


app = FastAPI()


class SkinTypeResponse(BaseModel):
    skin_type: str


@app.post("/macro")
async def predict_macro(file: UploadFile):
    async with aiofiles.open('image.jpg', 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    skin_type = predict_model.predict('image.jpg',
                                      'models/pigmented_nonpigmented_model.keras',
                                      'models/oily_dry_model.keras',
                                      'models/sensation_model.keras',
                                      'models/wrinkles_model.keras')
    response = SkinTypeResponse(
        skin_type=skin_type
    )
    return response
