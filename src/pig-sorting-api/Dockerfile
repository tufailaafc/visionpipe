FROM python:3.13-slim

WORKDIR /app


# Copy only the requirements file first (to leverage Docker's cache effectively)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt

################################## If this becomes large for dependencies uncomment to save bandwidth
# # has a lot of the pip dependencies already downloaded to save on bandwidth and time
# COPY ./packages /app/packages 

# COPY requirements.txt /app/

# # Download any missing packages from PyPI into /app/packages
# #RUN pip download -r requirements.txt --dest=/app/packages

# #Use the already downloaded dependencies first.
# RUN pip install --no-index --find-links=/app/packages -r requirements.txt --verbose 



COPY . /app/


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]