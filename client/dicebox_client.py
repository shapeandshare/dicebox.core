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

camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 10000);
camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 10000);


def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    # Our operations on the frame come here
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resized_im = resize_keep_aspect_ratio(im, 255)
    return im, resized_im


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
    camera_capture, resized_image  = get_image()
    filename = datetime.now().strftime('capture_%Y-%m-%d_%H_%M_%S_%f.png')

    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite('./tmp/%s' % filename, resized_image)

    with open('./tmp/%s' % filename, 'rb') as file:
        file_content = file.read()
    os.remove('./tmp/%s' % filename)
    #os.rename('./tmp/%s' % filename, './data/1d4/%s' % filename)
    base64_encoded_content = file_content.encode('base64')

    outjson = {}
    outjson['data'] = base64_encoded_content

    json_data = json.dumps(outjson)

    prediction = {}

    #print ('sending over the wire: %s' % json_data)
    headers = {
        'Content-type': 'application/json',
        'API-ACCESS-KEY': '6{t}*At&R;kbgl>Mr"K]=F+`EEe',
        'API-VERSION': '1.0.0'
    }

    try:
        #response = requests.post('https://dicebox.shapeandshare.com/api/prediction', data=json_data, headers=headers)
        #response = requests.post('http://172.16.0.79:5000/api/prediction', data=json_data, headers=headers)
        response = requests.post('http://127.0.0.1:5000/api/prediction', data=json_data, headers=headers)
        if response is not None:
            if response.status_code != 500:
                if 'prediction' in response.json():
                    prediction = response.json()['prediction']
                    print("%s" % prediction)
    except:
        print('.')
        #raise

    #print prediction

    #for key, value in sorted(prediction.iteritems(), key=lambda (k, v): (v, k)):
    #    #print "%s: %s" % (key, value)
    #    cv2.putText(camera_capture, "%s: %s" % (key, value), (5, 20 + 20 * i), font, 0.5, (255, 255, 255), 1)
    #    i -= 1
    #    if i < 0:
    #        i = 4

    cv2.namedWindow('dice box', cv2.WINDOW_NORMAL)
    # lets make a pretty output window

    #cv2.putText(camera_capture, "%s" % prediction, (20, 20), font, 0.5, (255, 255, 255), 1)
    #cv2.imshow('dice box', camera_capture)

    cv2.putText(resized_image, "%s" % prediction, (5, 15), font, 0.5, (255, 255, 255), 1)
    cv2.imshow('dice box', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


###############################################################################
# cleanup
###############################################################################
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
camera.release()
cv2.destroyAllWindows()
