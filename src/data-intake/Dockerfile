FROM python:3.13-slim

WORKDIR /app



RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt



COPY . /app/

#Not 100% decided which port we will have exposed
EXPOSE 8301 

CMD [ "python","NVRConnection.py" ]