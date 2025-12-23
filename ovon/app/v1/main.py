from fastapi import FastAPI
from ovon.app.v1.server import router as v1_router

app = FastAPI()
app.include_router(v1_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
