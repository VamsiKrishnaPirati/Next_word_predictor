from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Next Word Predictor",
    description="API for Next Word Prediction",
    version="1.0.0"
)

# Include API routes
app.include_router(router)
