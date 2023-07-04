# import kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import math
import cv2

class CamApp(App):
    def __init__(self, frame = None, ang = 0, *args):
        super(CamApp, self).__init__(*args)

        self.frame = frame
        self.wheel_img = cv2.imread('utils\\steering_wheel_image.jpg', 0)

        self.wheel_ang = ang
        self.rows,self.cols = self.wheel_img.shape
        self.smoothed_angle = 0

    def rotate_wheel(self):
        ### steering wheel visualization
        pred_angle = self.wheel_ang
        x = pred_angle
        delta = 1 - x ** 2
        if delta < 0: delta = 0
        y = math.sqrt(delta)
        degrees = 180 - ((math.atan2(y, x) * 180 / math.pi) - 90)
        self.smoothed_angle += 0.2 * pow(abs((degrees - self.smoothed_angle)), 2.0 / 3.0) * (degrees - self.smoothed_angle) / abs(
            degrees - self.smoothed_angle)
        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -self.smoothed_angle, 1)
        dst = cv2.warpAffine(self.wheel_img, M, (self.cols, self.rows))
        dst = cv2.flip(dst, -1)
        cv2.imwrite('utils\\steering_wheel_image - Copy.jpg',dst)

    def build(self):
        # object detect frame
        try:
            self.screen = Image(source=self.frame)
        except Exception as e:
            self.screen = Image()
            print(e)
            
        #steering wheel as frame
        self.wheel = Image(source ='utils\\steering_wheel_image - Copy.jpg')
        self.logo = Image(source='GUI\logo1.png')
        print(self.logo)
        layout = BoxLayout(orientation='vertical')

        self.logo.pos = (200, 0)
        self.logo.opacity = 1
        layout.add_widget(self.logo)
        layout.add_widget(self.screen)
        layout.add_widget(self.wheel)

        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    def update(self, dt):
        if self.wheel_ang != 0:
            self.rotate_wheel()
            self.wheel.reload()

        frame =self.frame
        if frame is not None:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture1.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            # display image from the texture
            self.screen.texture = texture1



