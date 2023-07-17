import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from utils.grabscreen import grab_screen
import math

# Load steering prediction model
model_name = 'steering\Models\steer_augmentation.h5'
model = load_model(model_name)

def steer_loop():
    """
    This function continuously captures screenshots of the screen, processes them, and predicts the steering angle and throttle using a pre-trained model.

    Yields:
    prediction (list): A list containing the predicted steering angle and throttle.

    Example:
    >>> for pred in steer_loop():
    >>>     print(pred)
    """
    # Countdown before starting the loop
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)
    
    while(1):
        # Capture screenshot of the screen 
        screen = grab_screen(region=(30,205,800,640))  

       # Preprocess the input image for the CNN 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160,120))

        # Make prediction using the pre-trained model
        prediction = model.predict([screen.reshape(-1,160,120,1)])[0]
        steering_angle = prediction[0]
        throttle = prediction[1]

        # Adjust the steering angle using math.atan
        steering_angle = math.atan(steering_angle*1.6)
        
        # Create a list with the adjusted steering angle and throttle
        prediction = [steering_angle,throttle]

        # Yield the prediction
        yield prediction
       
