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
        """
        Initializes the Driver class.

        Attributes:
        - steering_angle (Value): A shared value representing the steering angle.
        - throttle (Value): A shared value representing the throttle.
        - diag_len (Value): A shared value representing the diagonal length of detected objects.
        """
        self.steering_angle = Value('d', 0.0)
        self.throttle = Value('d', 0.0)
        self.diag_len = Value('d', 0.0)
        
    def steer_looper(self):
        """
        Continuously predicts the steering angle and controls the steering wheel simulation.

        The function acquires the shared steering_angle and throttle values, assigns the predicted values,
        and releases the acquired values. It also checks for the 'T' key press to pause the simulation.
        """
        paused = False
        for prediction in steer_loop():
            if paused == False:
                self.steering_angle.acquire()
                self.throttle.acquire()

                # Assign driving controls
                self.steering_angle.value ,pred_angle, self.throttle.value = prediction[0] ,prediction[0],  prediction[1]
    
                # Set the steering angle and throttle using the setJoy_Steer_Throttle_Brake function
                self.setJoy_Steer_Throttle_Brake( self.steering_angle.value, self.throttle.value,0)
                self.steering_angle.release()
                self.throttle.release()
   
            # Check for 'T' key press to pause the simulation
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
                
    def object_detector(self):
        """
        Continuously detects objects and updates the diag_len value.

        The function calls the detection_loop function to get the diagonal length of detected objects.
        If a diagonal length is obtained, it acquires the diag_len value, updates it, and releases it.
        If no diagonal length is obtained, it sets the diag_len value to 0.
        """
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
            
    def setJoy_Steer_Throttle_Brake (self,value_steerX, value_throttleX, brake_state,scale = 16384):
        """
        Sets the joystick position based on the steering angle, throttle, and brake state.

        The function takes the steering angle, throttle, and brake state as input and calculates the joystick position
        based on the provided scale. It generates the joystick position using the generateJoystickPosition function
        from the vjoy module, opens the joystick, updates the joystick position, and closes the joystick.

        Parameters:
        - value_steerX: The steering angle.
        - value_throttleX: The throttle.
        - brake_state: The brake state.
        - scale: The scale used for joystick position calculation. Default is 16384.
        """
        value_steerX = value_steerX + 1
        value_throttleX = value_throttleX + 1
        xPos_steering = int(value_steerX*scale)
        xPos_throttle = int(value_throttleX*scale)
        
        joystickPosition = vj.generateJoystickPosition(wAxisX = xPos_steering, wAxisY = scale,
                                            wAxisZRot = xPos_throttle, lButtons = brake_state)
        vj.open()
        vj.update(joystickPosition)
        vj.close()
        
    def pilot(self):
        """
        Controls the throttle and brake based on the detected object's diagonal length.

        The function continuously checks the throttle and brake values and adjusts them based on the
        detected object's diagonal length. If the diagonal length is greater than or equal to 150, it stops the car.
        If the diagonal length is between 50 and 150, it slows down the car. Otherwise, it drives normally.
        """
        #Clean input
        self.setJoy_Steer_Throttle_Brake(0,-1,0)
        while True:
            throttle = self.throttle.value
            brake = 0
            if self.diag_len.value != None:
                # Detected object is too close, STOP
                if self.diag_len.value >= 150:
                    throttle = -1.0
                    steering_angle = 0.0
                    brake = 1
                # Detected object is nearby, SLOW DOWN
                elif self.diag_len.value>=50 and self.diag_len.value<150:
                    if self.throttle.value>=0:
                        throttle = self.throttle.value/2
                    #throttle is below 0 i.e. deccelerate even more
                    else:
                        throttle= self.throttle.value -((1+self.throttle.value)/2)

            # Set the steering angle, throttle, and brake using the setJoy_Steer_Throttle_Brake function
            self.setJoy_Steer_Throttle_Brake( self.steering_angle.value, throttle, brake )
                
def main():
    """
    The main function that initializes the Driver class and starts the object detection process.

    It creates an instance of the Driver class and initializes the object detection process as a separate process.
    """
    driver = Driver()
    object_detection = Process(target = driver.object_detector)
    object_detection.start()

if __name__ == '__main__':
    main()



