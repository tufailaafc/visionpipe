## Brief overview of installation and use.

Instructions also found at: https://deeplabcut.github.io/DeepLabCut/docs/installation.html
2025/12/18

Prerequisets:
Conda
Linux(Ubuntu)
Terminal


# Installation:
```bash
conda create -n DEEPLABCUT python=3.12
conda activate DEEPLABCUT
conda install -c conda-forge pytables==3.8.0
```

If you want gpu support install pytorch for correct cuda version

```bash
pip3 install torch torchvision
```

Verify cuda compatability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```bash
pip install "deeplabcut[gui,tf]"
```
or just if you do not need tensor flow
```bash
pip install "deeplabcut[gui]"
```
^ This is the one I did - Nathaniel Yeo

May need to downgrade to: 
```bash
pip install pandas==2.1.4 - Nathaniel Yeo
```

Modules
DeepLabCutTest.py
This file has everything needed to start training a new model. From creating a new project, segemntation, labeling etc... <br>
Parts of the code can be commented out as they are only needed to be run once. 

DeepLabCutModelAnalysis.ipynb <br>
This is an evaluation script to show how the model reacts to different confidence thresholds and produce visuals, as well as annotated images of predictions...
variables you will need to switch.<br>
project_path: The path to your config.yaml file <br>
project_root_path: Likely the path to the folder containing your config.yaml file <br>
images_dir: The folder containing the images you wish to test with. <br>
output_dir: Where we store the annotated images and csv when saving evaluation of the model.


YOLOv11/data_splitter.py <br>
A simple script to split the dataset into train, validation, and test folders.

YOLOv11/DLCtoYOLOLabel.py <br>
Converts DLC style labels into YOLO style labels.

YOLOv11/PoseTrainingScript.ipynb <br>
This is a script that has code for training both yolo and deeplabcut models. It assumes you already have setput the projects and datasets.

YOLOv11/QuickLabelCheck.py <br>
A small script to verify that all of the labels have a matching image and vice versa for the yolo dataset.


YOLOv11/YOLO_Model_Evaluator.ipynb <br>
Evaluates a YOLO Pose model and gives per pixel error measurements to more easily compare with deeplabcut models.








