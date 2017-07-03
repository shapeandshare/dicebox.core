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
from lib import dicebox_config as config  # import our high level configuration

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


camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1000);
camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1000);

# 780x650
#camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1);
#camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1);

#60x50
#camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 60);
#camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 50);

def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, camera_capture = camera.read()
    # Our operations on the frame come here
    im = cv2.cvtColor(camera_capture, cv2.COLOR_BGR2GRAY)
    #resized_im = resize_keep_aspect_ratio(im, 255)

    height, width = im.shape[:2]

    extra_height = float(height) % 60
    height_offset = int(extra_height / 2)

    extra_width = float(width) % 50
    width_offset = int(extra_width / 2)

    # now crop
    x1 = int(0 + width_offset)
    y1 = int(0 + height_offset)
    x2 = int(width - width_offset)
    y2 = int(height - height_offset)
    #print("(%i, %i), (%i, %i)" % (y1, y2, x1, x2))
    cropped_image = im[y1:y2, x1:x2]


    resized_image = cv2.resize(cropped_image, (60, 50))
    # resized_image = cv2.resize(im, (60, 50))
    height, width = resized_image.shape[:2]

    height = float(height)
    width = float(width)
    #print("height: (%f)" % height)
    #print("width: (%f)" % width)

    return camera_capture, resized_image


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



with open("%s/category_map.txt" % config.DATA_DIRECTORY) as data_file:
    jdata = json.load(data_file)

server_category_map = {}
for d in jdata:
#    print(jdata[d])
#    print(d)
#    server_category_map[d] = jdata[d]
    #print("%s:%s" % (d, jdata[d]))
    server_category_map[d] = jdata[d]

CURRENT_EXPECTED_CATEGORY_INDEX=1
MAX_EXPECTED_CATEGORY_INDEX = len(server_category_map)
MISCLASSIFIED_CATEGORY_INDEX = True

KEEP_INPUT = False
ONLY_KEEP_MISCLASSIFIED_INPUT = True

SERVER_ERROR = False

#print(server_category_map)
###############################################################################
# main loop
###############################################################################
while (True):
    # Take the actual image we want to keep
    #camera_capture, resized_image  = get_image()
    camera_capture, resized_image = get_image()
    filename = datetime.now().strftime('capture_%Y-%m-%d_%H_%M_%S_%f.png')

    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite('./tmp/%s' % filename, resized_image)

    with open('./tmp/%s' % filename, 'rb') as file:
        file_content = file.read()

    if KEEP_INPUT and not SERVER_ERROR:
        #print(MISCLASSIFIED_CATEGORY_INDEX)
        #print(ONLY_KEEP_MISCLASSIFIED_INPUT)
        if not MISCLASSIFIED_CATEGORY_INDEX and ONLY_KEEP_MISCLASSIFIED_INPUT:
            os.remove('./tmp/%s' % filename)
        else:
            os.rename('./tmp/%s' % filename, './tmp/%s/%s' % (server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)], filename))
    else:
        os.remove('./tmp/%s' % filename)

    base64_encoded_content = file_content.encode('base64')

    outjson = {}
    outjson['data'] = base64_encoded_content

    json_data = json.dumps(outjson)

    prediction = {}
    category = {}
    #print ('sending over the wire: %s' % json_data)
    headers = {
        'Content-type': 'application/json',
        'API-ACCESS-KEY': '6{t}*At&R;kbgl>Mr"K]=F+`EEe',
        'API-VERSION': '0.1.0'
    }

    try:
        #response = requests.post('https://dicebox.shapeandshare.com/api/prediction', data=json_data, headers=headers)
        #response = requests.post('http://172.16.0.79:5000/api/prediction', data=json_data, headers=headers)
        response = requests.post('http://127.0.0.1:5000/api/prediction', data=json_data, headers=headers)
        if response is not None:
            if response.status_code != 500:
                SERVER_ERROR = False
                if 'prediction' in response.json():
                    prediction = response.json()['prediction']
                    category = server_category_map[str(prediction)]
                    #print("%s" % prediction)
                    #print("%s" % category)
    except:
        #print('.')
        SERVER_ERROR = True
        #raise

    #print prediction

    # training_category = '1d4_4'
    # if category == training_category:
    #     os.rename("./tmp/%s" % filename, "./tmp/%s/%s" % (category, filename))
    # else:
    #     print('misclassified')
    #     os.rename("./tmp/%s" % filename, "./tmp/misclassified/%s" % filename)


    #print(MAX_EXPECTED_CATEGORY_INDEX)
    #print(CURRENT_EXPECTED_CATEGORY_INDEX)
    #print(server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)])

    if category == server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)]:
        MISCLASSIFIED_CATEGORY_INDEX = False
        #print(False)
    else:
        MISCLASSIFIED_CATEGORY_INDEX = True
        #print(True)

    cv2.namedWindow('dice box', cv2.WINDOW_NORMAL)
    # lets make a pretty output window

    #cv2.putText(camera_capture, "%s" % prediction, (20, 20), font, 0.5, (255, 255, 255), 1)
    #cv2.imshow('dice box', camera_capture)

    #cv2.putText(camera_capture, "%s" % category, (5, 15), font, 0.5, (255, 255, 255), 1)

    # im[y1:y2, x1:x2]
    output_display = camera_capture
    height, width = output_display.shape[:2]
    output_display[height-50:height, 0:60] = cv2.cvtColor(resized_image, cv.CV_GRAY2RGB)

    output_label = "[pred %s/exp %s][match? %r][record? %r][only keep misclassified? %r][server error? %r]" % (category, server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)], not MISCLASSIFIED_CATEGORY_INDEX, KEEP_INPUT, ONLY_KEEP_MISCLASSIFIED_INPUT, SERVER_ERROR)

    cv2.putText(output_display, output_label, (5, 15), font, 0.3, (255, 255, 255), 1)

    cv2.imshow('dice box', output_display)

    input_key = cv2.waitKey(1)

    if input_key & 0xFF == ord('q'):
        break

    if input_key & 0xFF == ord('c'):
        if CURRENT_EXPECTED_CATEGORY_INDEX >= MAX_EXPECTED_CATEGORY_INDEX:
            CURRENT_EXPECTED_CATEGORY_INDEX = 1
        else:
            CURRENT_EXPECTED_CATEGORY_INDEX += 1

    if input_key & 0xFF == ord('z'):
        if KEEP_INPUT == True:
            KEEP_INPUT = False
        else:
            KEEP_INPUT = True

    if input_key & 0xFF == ord('b'):
            if ONLY_KEEP_MISCLASSIFIED_INPUT == True:
                ONLY_KEEP_MISCLASSIFIED_INPUT = False
            else:
                ONLY_KEEP_MISCLASSIFIED_INPUT = True

###############################################################################
# cleanup
###############################################################################
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
camera.release()
cv2.destroyAllWindows()
