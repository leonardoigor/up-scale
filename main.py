
import cv2
import numpy as np
from img_process import *
import fastapi
from PIL import Image


# img = cv2.imread('original.png')
# predicteds = Predict(img)
# paths = Save_temp(predicteds)
# print(predicteds.keys())

app = fastapi.FastAPI()


@app.post("/")
async def predict(img: fastapi. UploadFile = fastapi.File(...)):
    # convert to numpy array
    file = img.file
    new_img = np.fromstring(img.file.read(), np.uint8)
    image_pillow = Image.open(file)
    image_shape = image_pillow.size  # + (len(image_pillow.getbands()),)
    new_img = np.array(image_pillow)
    # fix image color
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    predicteds = Predict(new_img)

    paths = Save_temp(predicteds)
    return paths


@app.get("/img/{img_name:path}")
def get_img(img_name: str):
    return fastapi.responses.FileResponse(f"temp/{img_name}")


# route to scale image
@app.post("/scale")
def scale_img(width: int, height: int, file: fastapi. UploadFile = fastapi.File(...)):
    # convert to numpy array
    file = file.file
    image_pillow = Image.open(file)
    image_shape = image_pillow.size
    img = np.array(image_pillow)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    # return image
    path_save = f"temp/{width}x{height}.png"
    cv2.imwrite(path_save, img)
    return fastapi.responses.FileResponse(path_save)
