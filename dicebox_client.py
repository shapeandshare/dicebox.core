###############################################################################
# dice box
###############################################################################
import cv
import cv2
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


camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, config.IMAGE_WIDTH)
camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, config.IMAGE_HEIGHT)

# 780x650
# camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 1);
# camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 1);

# 60x50
# camera.set(cv.CV_CAP_PROP_FRAME_WIDTH, 60);
# camera.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 50);

font = cv.CV_FONT_HERSHEY_SIMPLEX

def get_image():
    retval, im = camera.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def resize_keep_aspect_ratio(input_image, desired_size):
    height, width = input_image.shape[:2]

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

    resized_input = cv2.resize(input_image, (new_width, new_height))

    output = numpy.zeros((desired_size, desired_size), numpy.uint8)

    x_offset = int(math.floor(x+new_width))
    y_offset = int(math.floor(y+new_height))
    # print("x_offset: (%i)" % x_offset)
    # print("y_offset: (%i)" % y_offset)

    # new lets drop the resized imput onto the output
    output[int(y):int(y_offset), int(x):int(x_offset)] = resized_input

    return output


# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(ramp_frames):
    temp = get_image()



with open('./category_map.json') as data_file:
    jdata = json.load(data_file)

server_category_map = {}
for d in jdata:
    # print(jdata[d])
    # print(d)
    # server_category_map[d] = jdata[d]
    # print("%s:%s" % (d, jdata[d]))
    # server_category_map[d] = jdata[d]
    server_category_map[str(jdata[d])] = str(d)

# print(server_category_map)

CURRENT_EXPECTED_CATEGORY_INDEX = 1
MAX_EXPECTED_CATEGORY_INDEX = len(server_category_map)
MISCLASSIFIED_CATEGORY_INDEX = True

KEEP_INPUT = False
ONLY_KEEP_MISCLASSIFIED_INPUT = True

SERVER_ERROR = False

###############################################################################
# main loop
###############################################################################
while True:
    # Take the actual image we want to keep
    # camera_capture, resized_image  = get_image()
    camera_capture = get_image()
    filename = datetime.now().strftime('capture_%Y-%m-%d_%H_%M_%S_%f.png')
    tmp_file_path = "%s/%s" % (config.TMP_DIR, filename)

    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite(tmp_file_path, camera_capture)

    with open(tmp_file_path, 'rb') as tmp_file:
        file_content = tmp_file.read()

    if KEEP_INPUT:
        # print(MISCLASSIFIED_CATEGORY_INDEX)
        # print(ONLY_KEEP_MISCLASSIFIED_INPUT)
        if not MISCLASSIFIED_CATEGORY_INDEX and ONLY_KEEP_MISCLASSIFIED_INPUT:
            os.remove(tmp_file_path)
        else:
            os.rename(tmp_file_path, '%s/%s/%s' % (config.TMP_DIR, server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)], filename))
    else:
        os.remove(tmp_file_path)

    base64_encoded_content = file_content.encode('base64')

    outjson = {}
    outjson['data'] = base64_encoded_content

    json_data = json.dumps(outjson)

    prediction = {}
    category = {}
    # print ('sending over the wire: %s' % json_data)
    headers = {
        'Content-type': 'application/json',
        'API-ACCESS-KEY': config.API_ACCESS_KEY,
        'API-VERSION': config.API_VERSION
    }

    try:
        url = "%s%s:%i/api/prediction" % (config.SERVER_URI, config.CLASSIFICATION_SERVER, config.SERVER_PORT)
        response = requests.post(url, data=json_data, headers=headers)
        if response is not None:
            if response.status_code != 500:
                SERVER_ERROR = False
                if 'prediction' in response.json():
                    prediction = response.json()['prediction']
                    category = server_category_map[str(prediction)]
                    # print("%s" % prediction)
                    # print("%s" % category)
    except:
        # print('.')
        SERVER_ERROR = True
        # raise

    # print prediction

    # training_category = '1d4_4'
    # if category == training_category:
    #     os.rename("./tmp/%s" % filename, "./tmp/%s/%s" % (category, filename))
    # else:
    #     print('misclassified')
    #     os.rename("./tmp/%s" % filename, "./tmp/misclassified/%s" % filename)

    # print(MAX_EXPECTED_CATEGORY_INDEX)
    # print(CURRENT_EXPECTED_CATEGORY_INDEX)
    # print(server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)])

    if category == server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)]:
        MISCLASSIFIED_CATEGORY_INDEX = False
        # print(False)
    else:
        MISCLASSIFIED_CATEGORY_INDEX = True
        # print(True)

    cv2.namedWindow('dice box', cv2.WINDOW_NORMAL)
    # lets make a pretty output window

    # cv2.putText(camera_capture, "%s" % prediction, (20, 20), font, 0.5, (255, 255, 255), 1)
    # cv2.imshow('dice box', camera_capture)

    # cv2.putText(camera_capture, "%s" % category, (5, 15), font, 0.5, (255, 255, 255), 1)

    # im[y1:y2, x1:x2]
    output_display = camera_capture
    # height, width = output_display.shape[:2]
    # output_display[height-50:height, 0:60] = cv2.cvtColor(resized_image, cv.CV_GRAY2RGB)

    output_label_1 = "[pred %s/exp %s][match? %r]" % (category, server_category_map[str(CURRENT_EXPECTED_CATEGORY_INDEX-1)], not MISCLASSIFIED_CATEGORY_INDEX)
    output_label_2 = "[record? %r][only keep misclassified? %r]" % (KEEP_INPUT, ONLY_KEEP_MISCLASSIFIED_INPUT)
    output_label_3 = "[server error? %r]" % SERVER_ERROR

    cv2.putText(output_display, output_label_1, (5, 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(output_display, output_label_2, (5, 35), font, 0.5, (255, 255, 255), 1)
    cv2.putText(output_display, output_label_3, (5, 55), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('dice box', output_display)

    input_key = cv2.waitKey(1)

    if input_key & 0xFF == ord('q'):
        break

    if input_key & 0xFF == ord('c'):
        KEEP_INPUT = False
        if CURRENT_EXPECTED_CATEGORY_INDEX >= MAX_EXPECTED_CATEGORY_INDEX:
            CURRENT_EXPECTED_CATEGORY_INDEX = 1
        else:
            CURRENT_EXPECTED_CATEGORY_INDEX += 1

    if input_key & 0xFF == ord('z'):
        if KEEP_INPUT is True:
            KEEP_INPUT = False
        else:
            KEEP_INPUT = True

    if input_key & 0xFF == ord('b'):
            if ONLY_KEEP_MISCLASSIFIED_INPUT is True:
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
