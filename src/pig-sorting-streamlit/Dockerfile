FROM python:3.13-slim

WORKDIR /app


# RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (to leverage Docker's cache effectively)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt


COPY . /app/


EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
