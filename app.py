import os
from logging import getLogger
from fastapi import FastAPI, Response, status, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from clip import Clip, ClipInput
from meta import Meta
import pytesseract
from PIL import Image
import io
import whisper

clip: Clip
meta_config: Meta
logger = getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip
    global meta_config

    cuda_env = os.getenv("ENABLE_CUDA")
    cuda_support = False
    cuda_core = ""

    if cuda_env is not None and cuda_env == "true" or cuda_env == "1":
        cuda_support = True
        cuda_core = os.getenv("CUDA_CORE")
        if cuda_core is None or cuda_core == "":
            cuda_core = "cuda:0"
        logger.info(f"CUDA_CORE set to {cuda_core}")
    else:
        logger.info("Running on CPU")

    clip = Clip(cuda_support, cuda_core)
    meta_config = Meta()
    logger.info("Model initialization complete")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
async def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
async def meta():
    return await meta_config.get()


@app.post("/vectorize")
async def read_item(payload: ClipInput, response: Response):
    try:
        result = await clip.vectorize(payload)
        return {
            "textVectors": result.text_vectors,
            "imageVectors": result.image_vectors,
        }
    except Exception as e:
        logger.exception("Something went wrong while vectorizing data.")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}


@app.post("/ocr")
async def ocr_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename.lower()
    try:
        # Only support images -- client splits PDF pages already
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
            image = Image.open(io.BytesIO(contents))
            text = pytesseract.image_to_string(image)
            return text
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only image files are supported.",
            )
    except Exception as e:
        logger.exception("Error during OCR processing.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Save to a temporary file for whisper
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as temp_audio:
            temp_audio.write(contents)
            temp_audio_path = temp_audio.name
        # Load whisper model (base for speed, can be changed)
        model = whisper.load_model("base")
        result = model.transcribe(temp_audio_path)
        os.remove(temp_audio_path)
        return {"text": result["text"]}
    except Exception as e:
        logger.exception("Error during audio transcription.")
        raise HTTPException(status_code=500, detail=str(e))
