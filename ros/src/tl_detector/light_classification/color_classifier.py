from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image, ImageOps
import os
import rospy
import tensorflow as tf

#from traffic_light_colors import TrafficLight#enum in traffi clight colours was a problem in the test
from styx_msgs.msg import TrafficLight # for traffic light colour values

class ColorClassifier:
    def __init__(self, is_site):
        self.IMAGE_SIZE = 32
        cwd = os.path.dirname(os.path.realpath(__file__))
        # use different model for simulator and parking lot
        if is_site:
            self.class_model = load_model(cwd + '/models/model.h5')
        else:
              self.class_model = load_model(cwd + '/models/model_sim.h5')
        self.class_graph = tf.get_default_graph()

    def predict_image(self, image):
        x = self.crop_image(image)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        with self.class_graph.as_default():
            preds = self.class_model.predict_classes(x)
            prob = self.class_model.predict_proba(x)

        return preds[0], prob[0]
        #return TrafficLight.RED, prob[0]# i am making it alway return RED as a test

    def predict_images(self, images):
        predictions = []
        for image in images:
            pred, prob = self.predict_image(image)
            #rospy.loginfo('Color pred {} with prob'.format(pred, prob))
            rospy.loginfo('Color pred {} with prob {}'.format(pred, prob)) #john added {} here missing?
            predictions.append(pred)
            # added by John to force a return on detection of even one red
            if (pred == 0): return TrafficLight.RED
            
            if len(predictions) > 0:
                prediction = max(predictions, key=predictions.count)
                if prediction == 3:
                    return TrafficLight.UNKNOWN
                else:
                    return prediction
        else:
            return TrafficLight.UNKNOWN

    def crop_image(self, img):
        img.thumbnail((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.ANTIALIAS)
        width, height = img.size
        delta_w = self.IMAGE_SIZE - width
        delta_h = self.IMAGE_SIZE - height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding, fill=0)  # fill with black dots

        return img

