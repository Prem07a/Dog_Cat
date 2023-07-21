from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="code/website/templates")
app.mount("/static", StaticFiles(directory="code/website/static"), name="static")  # Serve the static folder

MODEL = tf.keras.models.load_model("models/model_train_7")
CLASS_NAMES = ['CAT', 'DOG']

def read_file_as_image(data) -> np.ndarray:
    """
    Reads the uploaded file data and converts it into a numpy array (image representation).

    Args:
        data (bytes): The raw bytes of the uploaded file.

    Returns:
        np.ndarray: The image data as a numpy array.
    """
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main page (index.html) of the Image Classification web application.

    Args:
        request (Request): The FastAPI Request object.

    Returns:
        HTMLResponse: The rendered HTML template response for the main page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Handles the image upload and prediction process.

    Args:
        request (Request): The FastAPI Request object.
        file (UploadFile): The uploaded file object representing the image.

    Returns:
        HTMLResponse: The rendered HTML template response displaying the prediction result.
    """
    image = read_file_as_image(await file.read())

    # Resize the image to (80, 80) as expected by the model
    image = tf.image.resize(image, (80, 80))
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = ((np.max(predictions[0]) * 100 * 100) // 1) / 100

    # Convert the image to base64 to display it in the HTML page
    image_pil = Image.fromarray(np.uint8(image))
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return templates.TemplateResponse("index.html", {"request": request, "result": {"class": predicted_class, "confidence": confidence, "image_base64": image_base64}})

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="localhost", port=8000, reload=True)
