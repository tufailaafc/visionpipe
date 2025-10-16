FROM python:3.13-slim

WORKDIR /app


# Helps with running grapical programs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt



COPY . /app/

EXPOSE 8801 

CMD [ "python","StreamConversionServer.py" ]