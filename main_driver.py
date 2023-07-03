### imoprts
from multiprocessing import Process , Value
import threading
import sys
import atexit
import time
import math
import cv2
# custom modules
from steering.vjoy import vj, setJoy 
from steering.driving_prediction import steer_loop
from object_detection.ssd_mobilenet import detection_loop
from utils.getkeys import key_check


class Driver:
    def __init__(self):
        self.steering_angle = Value('d', 0.0)
        self.throttle = Value('d', 0.0)
        self.diag_len = Value('d', 0.0)
        
    ### steering prediction and steering wheel simulation    
    def steer_looper(self):
        paused = False
        for prediction in steer_loop():
            if paused == False:
                self.steering_angle.acquire()
                self.throttle.acquire()

                # assign driving controls
                self.steering_angle.value ,pred_angle, self.throttle.value = prediction[0] ,prediction[0],  prediction[1]
    
                # local variable pred_angle was assigned out of the unnecessaty of using the class member,
                # using locals is faster
                self.setJoy_Steer_Throttle_Brake( self.steering_angle.value, self.throttle.value,0)
                self.steering_angle.release()
                self.throttle.release()
   
            ### press T to pause the simulation
            keys = key_check()
            if 'T' in keys:
                if paused ==True:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    self.steering_angle.value = 0
                    self.throttle.value = -1
                    brake = 0
                    self.setJoy_Steer_Throttle_Brake(self.steering_angle.value,self.throttle.value, brake)
                    time.sleep(1)
                
    ### receives diagonal lengths of detected objects       
    def object_detector(self):
        while True:
            diag_len = detection_loop(self.steering_angle.value)
            if diag_len is not None:
                self.diag_len.acquire()
                self.diag_len.value = diag_len
                self.diag_len.release()
            else:
                self.diag_len.acquire()
                self.diag_len.value = 0
                self.diag_len.release() 
            

    
    ### Joystick car input feeder 
    def setJoy_Steer_Throttle_Brake (self,value_steerX, value_throttleX, brake_state,scale = 16384):
        value_steerX = value_steerX +1
        value_throttleX = value_throttleX +1
        xPos_steering = int(value_steerX*scale)
        xPos_throttle = int(value_throttleX*scale)
        
        ''' adding a manual bias since the model lacks right turns '''
        #if xPos_steering < scale:
            #xPos_steering = int(xPos_steering*1.5)
            
        joystickPosition = vj.generateJoystickPosition(wAxisX = xPos_steering, wAxisY = scale,
                                            wAxisZRot = xPos_throttle, lButtons = brake_state)
        vj.open()
        vj.update(joystickPosition)
        vj.close()
        
    def pilot(self):
        #makes sure the joystick input is clean
        self.setJoy_Steer_Throttle_Brake(0,-1,0)
        while True:
            throttle = self.throttle.value
            brake = 0
            #print(self.diag_len.value)
            #self.diag_len.acquire()
            if self.diag_len.value != None:
                #print("diaginal len:", self.diag_len.value)
                # Detected object is too close so STOP
                if self.diag_len.value >= 150:
                    #print("stop")
                    throttle = -1.0
                    steering_angle = 0.0
                    brake = 1
                # Detected object is nearby so SLOW DOWN
                elif self.diag_len.value>=50 and self.diag_len.value<150:
                    #print("slow")
                    if self.throttle.value>=0:
                        throttle = self.throttle.value/2
                    #throttle is below 0 i.e. deccelerate even more
                    else:
                        throttle= self.throttle.value -((1+self.throttle.value)/2)
                '''
                else: #Detected object is small or 0
                    #print("drive normal")
                    if self.diag_len.value > 0 :
                        if self.throttle.value >=0:
                            throttle = self.throttle.value /2
                        else:# throttle is below 0
                            throttle = self.throttle.value-((1+self.throttle.value)/2)
                    elif self.throttle.value = -1: # detected object is 0 and car is not moving
                        throttle = self.throttle.value-((1+self.throttle.value)/2)
                '''
                    
                    
         
            #if self.throttle.value <0.5:
                #throttle = self.throttle.value+1
            self.setJoy_Steer_Throttle_Brake( self.steering_angle.value, throttle, brake )
            #self.diag_len.value  = 0
                
def main():
    driver = Driver()
    # initialize processes
    #steering_set = Process(target = driver.steer_looper)
    object_detection = Process(target = driver.object_detector)
    #pilot = Process(target = driver.pilot)

    # start processes
    #steering_set.start()
    object_detection.start()
    #pilot.start()

if __name__ == '__main__':
    main()



