# Protodrive
![logo](https://github.com/RanelDL/Protodrive/assets/61747694/654a4767-72a8-4caa-8d17-9b0354bb16cc)

Simple self-driving car simulation, final project of Software Engineering class, HS senior year.

## Instructions:
    1. Run "pip install -r requirements.txt" in cmd.
    2. Find a pov video of a car driving on a road. 
    3. Run main_driver.py.

## Development Process:
### Key goals & requirements:
* Identify the road correctly and keep the vehicle in its travel lane.
* Return an acceleration value that will match the driving conditions (lower acceleration in turns and higher when driving on a straight track).
* Identify objects listed in the list known to the system (car, passerby, bus, motorcycle, train) and alert the driver about them / automatically avoid them (detecting an approaching vehicle will cause the system to slow down).
* Graphically present a virtual steering wheel controlled by the software that predicts the vehicle's journey.

### Chosen Algorithms:
Object detection: SSD MobileNet
* Mobile-friendly, chosen for its speed and efficiency.

Steering: Nvidia's PilotNet
* The model is of a discriminative type, focusing on classifying inputs into output values, as opposed to generative models that create new examples after learning the database.
* The neural network, specifically a Convolutional Neural Network (CNN), is selected because it can automatically identify important features in images through convolution layers. The CNN utilizes different-sized filters to detect various features, starting from simple patterns and progressing to more complex objects.
* Selected for its balanced structure and ease of parameter control.

The implementation utilizes TensorFlow, Keras, and OpenCV libraries for machine learning and image processing.
