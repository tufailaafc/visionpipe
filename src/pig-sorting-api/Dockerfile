FROM python:3.13-slim

WORKDIR /app


# Install system packages required by OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (to leverage Docker's cache effectively)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt && pip freeze

################################## If this becomes large for dependencies uncomment to save bandwidth
# # has a lot of the pip dependencies already downloaded to save on bandwidth and time
# COPY ./packages /app/packages 

# COPY requirements.txt /app/

# # Download any missing packages from PyPI into /app/packages
# #RUN pip download -r requirements.txt --dest=/app/packages

# #Use the already downloaded dependencies first.
# RUN pip install --no-index --find-links=/app/packages -r requirements.txt --verbose 



COPY . /app/


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]