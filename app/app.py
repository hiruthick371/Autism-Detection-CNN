from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from .model import load_model, predict_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    prediction = predict_image(model, image)
    result = "Autism" if prediction == 0 else "Non-Autism"
    return JSONResponse({"prediction": result})
