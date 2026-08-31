FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app \
    HF_HUB_OFFLINE=1

RUN groupadd -r doserad && useradd -m --no-log-init -r -g doserad doserad
WORKDIR /opt/app

COPY docker/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY doserad2026 /opt/app/doserad2026
COPY scripts/predict_sct_public.py /opt/app/scripts/predict_sct_public.py
COPY docker/app.py docker/inference.py /opt/app/

RUN chown -R doserad:doserad /opt/app
USER doserad

LABEL org.grand-challenge.api-method="invoke" \
      org.opencontainers.image.source="https://github.com/HEMONCedric/DoseRAD2026"

EXPOSE 4743
ENTRYPOINT ["python", "/opt/app/app.py"]
