# FastAPI app

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from paths import classify, register
from tensorflow.keras.models import load_model
from start_up_functions import check_models

app = FastAPI()


class ImgLoc(BaseModel):
    image_path: str = Field(
        description="URL for the images in folder")

    def __getitem__(self, item):
        print(item)
        return getattr(self, item)


@app.on_event("startup")
async def startup_event():
    if not check_models():
        # register()
        pass
    else:
        print('Model already exists, using existing model')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/classify")
def classify_image(image_location: ImgLoc = Body(embed=True)):
    # print(image_location['image_path'] + ' is the image location \n')
    return classify.main(image_location['image_path'])
    # return classify.main()


@app.post("/register")
def register_image(image_location: ImgLoc = Body(embed=True)):
    return register.main(image_location['image_path'])
