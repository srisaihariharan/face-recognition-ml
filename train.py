import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

# -------------- image labels ------------------------
def getImagesAndLabels(path):
    """
    Get image paths and corresponding labels from directory
    """
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empty face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

# ----------- train images function ---------------
def TrainImages():
    """
    Train LBPH face recognizer from training images
    """
    # Create LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    
    # Get faces and labels from training image folder
    faces, Id = getImagesAndLabels("TrainingImage")
    
    # Train the recognizer with threading for non-blocking operation
    Thread(target = recognizer.train(faces, np.array(Id))).start()
    
    # Display counter during training (optional)
    Thread(target = counter_img("TrainingImage")).start()
    
    # Save trained model
    recognizer.save("TrainingImageLabel"+os.sep+"Trainner.yml")
    print("Training Completed!")

# Optional, adds a counter for images trained
def counter_img(path):
    """
    Display counter of images being trained
    """
    imgcounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        print(str(imgcounter) + " Images Trained", end="\r")
        time.sleep(0.008)
        imgcounter += 1

if __name__ == '__main__':
    print("Starting training...")
    TrainImages()
    print("\nTraining finished successfully!")
