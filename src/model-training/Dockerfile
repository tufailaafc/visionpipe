# Has all the requirements for ultralytics including gpu support
FROM ultralytics/ultralytics:latest 



# Set working directory to /app
WORKDIR /app


# has a lot of the pip dependencies already downloaded to save on bandwidth and time
COPY ./packages /app/packages 

COPY requirements.txt /app/

# Download any missing packages from PyPI into /app/packages
#RUN pip download -r requirements.txt --dest=/app/packages

#Use the already downloaded dependencies first.
RUN pip install --no-index --find-links=/app/packages -r requirements.txt --verbose 



# Copy the rest of the application
COPY . /app/



# Expose the port and run the application, ensure the logs are not suppressed
CMD ["uvicorn", "trainer:app", "--host", "0.0.0.0", "--port", "8401", "--log-level", "info"]
