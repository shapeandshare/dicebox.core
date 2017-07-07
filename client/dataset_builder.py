###############################################################################
# dice box
###############################################################################
import cv, cv2
from datetime import datetime
import json
import requests
import os
import numpy
import math

###############################################################################
# configure our camera, and begin our capture and prediction loop
###############################################################################
# Camera 0 is the integrated web cam on my netbook
camera_port = 0

# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 3

# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1920);
camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1080);


def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    # Our operations on the frame come here
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #resized_im = resize_keep_aspect_ratio(im, 255)
    return im #, resized_im


def resize_keep_aspect_ratio(input, desired_size):
    height, width = input.shape[:2]

    height = float(height)
    width = float(width)
    # print("height: (%f)" % height)
    # print("width: (%f)" % width)

    if width >= height:
        max_dim = width
    else:
        max_dim = height

    scale = float(desired_size) / max_dim

    if width >= height:
        new_width = desired_size
        x = 0
        new_height = height * scale
        y = (desired_size - new_height) / 2
    else:
        y = 0
        new_height = desired_size
        new_width = width * scale
        x = (desired_size - new_width) / 2


    # print("desired size: (%f)" % desired_size)
    # print("x: (%f)" % x)
    # print("y: (%f)" % y)

    new_height = int(math.floor(new_height))
    new_width = int((math.floor(new_width)))
    # print("new_height: (%i)" % new_height)
    # print("new_width: (%i)" % new_width)

    resized_input = cv2.resize(input, (new_width, new_height))

    output = numpy.zeros((desired_size, desired_size), numpy.uint8)

    x_offset = int(math.floor(x+new_width))
    y_offset = int(math.floor(y+new_height))
    #print("x_offset: (%i)" % x_offset)
    #print("y_offset: (%i)" % y_offset)

    # new lets drop the resized imput onto the output
    output[int(y):int(y_offset), int(x):int(x_offset)] = resized_input

    return output


# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(ramp_frames):
    temp = get_image()

font = cv.CV_FONT_HERSHEY_SIMPLEX


###############################################################################
# main loop
###############################################################################
while (True):
    # Take the actual image we want to keep
    camera_capture = get_image()
    filename = datetime.now().strftime('capture_%Y-%m-%d_%H_%M_%S_%f.png')

    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite('./tmp/%s' % filename, camera_capture)

    #with open('./tmp/%s' % filename, 'rb') as file:
    #    file_content = file.read()
    #os.remove('./tmp/%s' % filename)
    #os.rename('./tmp/%s' % filename, './data/1d4/%s' % filename)

    cv2.namedWindow('dice box', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('(resized) dice box', cv2.WINDOW_NORMAL)
    cv2.imshow('dice box', camera_capture)
    #cv2.imshow('(resized) dice box', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


###############################################################################
# cleanup
###############################################################################
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
camera.release()
cv2.destroyAllWindows()
