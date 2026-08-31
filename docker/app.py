"""Grand Challenge invoke server for the public Proton pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
import uvicorn
from uvicorn.config import LOGGING_CONFIG

import inference
from doserad2026.runtime import ProtonRuntime


RUNTIME: ProtonRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUNTIME
    RUNTIME = ProtonRuntime()
    RUNTIME.load()
    yield
    RUNTIME = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    code = status.HTTP_200_OK if RUNTIME is not None and RUNTIME.ready else status.HTTP_404_NOT_FOUND
    return Response(status_code=code)


@app.post("/invoke")
async def invoke():
    if RUNTIME is None:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    inference.run(RUNTIME)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    config = LOGGING_CONFIG.copy()
    config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    uvicorn.run(app, host="0.0.0.0", port=4743, log_config=config)
