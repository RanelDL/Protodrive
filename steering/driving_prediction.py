import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
from utils.grabscreen import grab_screen
import math

### load steering prediction model
model_name = 'steering\Models\steer_augmentation.h5'
model = load_model(model_name)

def steer_loop():
    for i in list(range(2))[::-1]:
        print(i+1)
        time.sleep(1)
    
    while(1):
        # screenshot of the screen 
        screen = grab_screen(region=(30,205,800,640))  
        '''-------------------resize and reshape the input image for CNN----------------'''
        
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160,120))
        prediction = model.predict([screen.reshape(-1,160,120,1)])[0]
        #  steering_angle is prediction [0],
        #  throttle is prediction [1]

        ''' multiplication by 1.5'''
        #steering_angle = prediction[0]
        #throttle = prediction[1]

        # amplify prediction angle by a constant factor of 1.5
        #if steering_angle > 0.20 or steering_angle <-0.20:
            #steering_angle = steering_angle*1.5
        #prediction = [steering_angle,throttle]
        
        ''' red log func'''  #  too aggressive
        '''
        steering_angle = prediction[0]
        throttle = prediction[1]
        print("before:" , steering_angle)
        # angle < 0
        if steering_angle < 0:
            steering_angle =  math.log((2.3*abs(steering_angle))+0.25,10) +0.6
            if steering_angle < -1 :
                steering_angle = -1 # round values bigger than threshold
            steering_angle = -1*steering_angle
            print("after: ", steering_angle)
            
        # angle >= 0    
        else:
            steering_angle = math.log((2.3*abs(steering_angle))+0.25,10) +0.6
            if steering_angle > 1 :
                steering_angle = 1
        '''
        # # # # # # # # # # # # # # # #
        
        ''' custom black arctan '''
        '''
        steering_angle = prediction[0]
        throttle = prediction[1]

       
        steering_angle = 0.69 * math.atan(8*steering_angle)
        if abs(steering_angle) > 1:
            steering_angle = 1

        '''

        ''' atan x '''
        steering_angle = prediction[0]
        throttle = prediction[1]
        steering_angle = math.atan(steering_angle*1.6)
        
        prediction = [steering_angle,throttle]
        yield prediction
       
