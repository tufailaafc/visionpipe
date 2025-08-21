# Purpose
This is a project that aims to help manage, train models and annotate images.

## Default ports
mongo(database) = 27017
pig-sorting-api(backend) = 8001
LabelStudio(annotation) = 8080
data-intake(data ingestion) = 8301
model-training = 8401
pig-sorting-streamlit(frontend) = 8501
model-deploy = 8601
deep-stream(app) = 8701
deep-stream(rtsp output WIP) = 8554
data-stream(WIP) = 8801



# What it does, break down of components
## data-intake (Not fully implemented)
The data-intake container's purpose is to ingest data from sensors and the nvr.
It then does some level of processing like ensuring that all of the meta data is correct.
It will then store the meta data in the MongoDB and the images and videos in data/images and data/videos.

### Troubleshooting 
wsse error: If you have an error that has someting to do with the wsse times not being correct then you should:
Ensure you are on the same network and as the NVR and enter the ip into your browser.
Then login with the NVR credentials.
Select System.
Select Date&Time
Make sure the timzone is correct.
Select Sync PC.
Then click apply.

## data-processor (Not yet implemented)

## data-stream (WIP)
This container will take in a rtsp stream and output an mjpeg stream for easier consumption by the streamlit frontend

## deep-stream (Not fully implemented yet)
This container takes in a rtsp stream perform some inference on it and save the video to a file.
Currently has some weird behaviour, in order to properly save the file so you can view it you must 
properly end the stream, easiest way that I have found was to switch my internet connection so that it 
drops. It will also currently overwrite the video file.
TODO: 
    Make it stream out an rtsp stream that will hook into datastream.
    Ensure that you can add multiple rtsp streams. 
    Potentailly add some logging to mongoDB.
    Add secondary inference engines.
    Add trackers for the objects detected in inference.

## deployment (Not yet implemented)
This module will likely handle any deployment configuration when sending things to the cloud

## LabelStudio (Not fully implemented)
We are using this as our primary annotation software. It will take the data from our MongoDB and then output the annotations to data/label-studio-data right now. The interface can be accessed at localhost:8080

TODO: 
    Make it export properly to the processed folder.

When mounting any folder you will need to change the ownership to 1001.
There are currently two folders which this should be run on.

```bash
sudo chown 1001 /data/label-studio-data
```

```bash
sudo chown 1001 processed
```


## model-deploy (WIP)
Loads the models trained by the model training container and will allow the user to upload an image and perform inference with the selected model.
Currently has the ability to dynamically select the confidence threshold.
TODO:
    Might want to change it to use a triton inference server.


## model-training (Not fully implemented)
This will handle the training and evaluation of the models.

What it requires: 
    A path to the model. If you put in any of the standard yolo models it will download them for you. For example yolo11n.pt
    A path to the dataset. Specfically the .yaml file. Must be accesible inside of the docker container. Will auto find if placed in data/processed and should be a .zip folder currently.
    A project name. this will be the top level folder.
    A run name. This will be for the specific run in the project.
It will save the run and the models in the model folder. This will also have all of the validation metrics.
It will send the metrics to pig-sorting-api which will log the data in mongoDB

TODO: 
    Make sure that all training parameters are saved with the run.

### Setup for GPU support REQUIRED!!!
Follow this guide for setting up gpu access https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

This will allow the container access to the gpu.





## pig-sorting-api(Not fully implemented)
This is our primairy backend for our services and will handle routing between all of the different modules. It's primary role will be to handle routing data from the frontend to the different modules.
And from the different modules to the databse.


## pig-sorting-streamlit(Not fully implemented)
This will be our primairy frontend and will allow the user to interact with our system.
TODO
    It will allow the user to query our MongoDB and generate graphs etc...
    It will also have some way to talk to the data-intake module to schedule pictures, videos etc...
    Add some form of authentication and authorization to limit user access.

Current pages:
    DataBaseView:
        Allows the user to see the captured images and the metadata
    DeepStreamView:
        (WIP) Will allow the user to add extra rtsp sources and should also allow the user to see the output stream
    ModelDeployView:
        Has options to select trained models and perform inference with it.
    ModelTrainView:
        Allows the user to configure some training parameters(should be increased later), and select datasets, base model to train from, and displays some of the training data in realtime.



## workflow(Not yet implemented)
Some form of automated testing and ci/cd.



# Requirements
Should be all handled by running sudo docker compose up --build
Requires opencv-python==4.10.0.84 newer versions will cause timeout error.

You may have issues running some containers as they have some variation of:
```bash
COPY ./packages /app/packages 

RUN pip install --no-index --find-links=/app/packages -r requirements.txt --verbose 
```
This is done to reduce how much bandwidth is used if rebuilding often.

You can run this to fix that issue or switch the pip install command to download instead:
```bash
pip download -r requirements.txt -d ./packages
```





# Setup
For running Label Studio you will need to run when inside src in terminal:
```bash
sudo chown -R 1001:1001 ../data/label-studio-data
```
This will allow the container to write to the folder.

After that run 

```bash
sudo docker compose up --build
```

May also require the user to allow legacy tokens to allow exporting with images.


# secrets
In order for everything to start up and work correctly you will need to create a .env file inside of src.
The inside of the .env file should look something like this.

```bash
MONGO_INITDB_ROOT_USERNAME=yourUsername
MONGO_INITDB_ROOT_PASSWORD=yourPassword

LABEL_STUDIO_USERNAME=your@gmail.com
LABEL_STUDIO_PASSWORD=yourOtherPassword

NVR_USERNAME=theUserNameForTheNvr
NVR_PASSWORD=thePasswordToTheNvr
NVR_IP=192.168.1.5
NVR_PORT=554
```

The other secret file goes in src/pig-sorting-api/.streamlit/secrets.toml
And it should look like this.
```bash
[mongo]
host = "mongo"
port = 27017
username = "yourMongoUserName"
password = "yourMongoPassword"
```


