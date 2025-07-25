FROM python:slim3.13

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt



COPY . /app/

#Not 100% decided which port we will have exposed
EXPOSE 8301 

CMD [ "NVRConnection.py" ]