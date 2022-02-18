# Dockerfile (for Trainer)
FROM python:3.9-slim

ENV PATH_APP /app
WORKDIR $PATH_APP

# ENV SERVER_IP 127.0.0.1
# ENV SERVER_PORT 65431
# EMV EPOCHS 30

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./worker_training.py" ]