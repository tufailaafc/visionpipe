# Purpose
This is a project that aims to help manage, train models and annotate images.


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

## deployment (Not yet implemented)
This module will likely handle any deployment configuration when sending things to the cloud

## LabelStudio (Not fully implemented)
We are using this as our primary annotation software. It will take the data from our MongoDB and then output the annotations to data/label-studio-data right now. The interface can be accessed at localhost:8080

When mounting any folder you will need to change the ownership to 1001.
There are currently two folders which this should be run on.

```bash
sudo chown 1001 /data/label-studio-data
```

```bash
sudo chown 1001 processed
```


## model-deploy (Not yet implemented)
This will handle deploying the already trained models to wherever we need them.

## model-training (Not yet implemented)
This will handle the training and evaluation of the models

## pig-sorting-api(Not fully implemented)
This is our primairy backend for our services and will handle routing between all of the different modules.


## pig-sorting-api(Not fully implemented)
This will be our primairy frontend and will allow the user to interact with our system.
It will allow the user to query our MongoDB and generate graphs etc...
It will also have some way to talk to the data-intake module to schedule pictures, videos etc...
It will allow the testing and selection of models as well.

## workflow(Not yet implemented)
Some form of automated testing and ci/cd.



# Requirements
Should be all handled by running sudo docker compose up --build
Requires opencv-python==4.10.0.84 newer versions will cause timeout error.




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

