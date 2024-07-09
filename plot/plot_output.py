import numpy as np
import cv2

def plot_output(predict, W, H, path):

    predict = predict.reshape([W, H])
    color_image = np.zeros((W, H, 3)) 

    color_image[predict == 0] = (0, 0, 255)  # Red
    color_image[predict == 1] = (0, 255, 0)  # Green
    print(path)
    cv2.imwrite(path + ".png", color_image)

