FROM python:3.10-slim

ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Mock model download step from MLflow using the run id.
RUN echo "Downloading model artifacts for RUN_ID=${RUN_ID}" > /app/model_download.log

CMD ["sh", "-c", "echo Container started with model from RUN_ID=${RUN_ID}; cat /app/model_download.log"]
