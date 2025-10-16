FROM python:3.13-slim

WORKDIR /app



RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ffmpeg \
    # For debugging, not needed for deployment
    curl \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt



COPY . /app/


EXPOSE 8301 

CMD ["uvicorn", "NVRConnection:app", "--host", "0.0.0.0", "--port", "8301", "--log-level", "info"]