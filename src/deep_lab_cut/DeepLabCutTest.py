import deeplabcut

# This code is only needed if starting a new project
######################################

# The videos from which we will annotate for our new project
VIDEO_DIRECTORY = "/home/engrkhan/visionpipe/data/DeepLabCut"

# The main directory for our new project.
WORKING_DIRECTORY = "/home/engrkhan/visionpipe/src/deep_lab_cut"

# This will build a project with the correct folder structure and create symbolic links for the video 
# deeplabcut.create_new_project('PorkAnalysis', 'WG', 
#                               videos=[VIDEO_DIRECTORY],
#                               working_directory=WORKING_DIRECTORY)

#######################################

# This is the path to our primary configuration file config.yaml which is where most project settings will be found.
# This includes the list of body parts as well as the skeleton

config_path="/home/engrkhan/visionpipe/src/deep_lab_cut/PorkAnalysis-WG-2026-06-08/config.yaml"

# # Inputs are needed to ensure we wait for the process to be done before continuing
# input("Press Enter to continue to extraction...")
# # We extract frames from the videos to annotate
# deeplabcut.extract_frames(config_path, mode="automatic")


input("Press Enter to continue to label...")
# creates or adds labels to the images
deeplabcut.label_frames(config_path)


input("Press Enter to continue check labels...")
# Ensure that we are happy with all of the labels
deeplabcut.check_labels(config_path)


input("Press Enter to continue and create the training dataset...")
# Generate the dataset along with the .yaml training file.
deeplabcut.create_training_dataset(config_path, num_shuffles=1)


input("Press Enter to continue and train the network on the dataset...")
deeplabcut.train_network(config_path)


