import cv, cv2
from datetime import datetime
import json
import requests
import os
import numpy
import math
from lib import dicebox_config as config  # import our high level configuration
from lib import filesystem_connecter # inport our file system connector for input
#import tensorflow as tf # for tensorflow
import json # for writing category data to file
import operator

###############################################################################
# prep our data sets
###############################################################################
network_input_index = filesystem_connecter.get_data_set(config.DATA_DIRECTORY)
category_map = filesystem_connecter.get_data_set_categories(network_input_index)

with open("%s/category_map.txt" % config.DATA_DIRECTORY) as data_file:
    jdata = json.load(data_file)

server_category_map = {}
for d in jdata:
#    print(jdata[d])
#    print(d)
#    server_category_map[d] = jdata[d]
    #print("%s:%s" % (d, jdata[d]))
    server_category_map[d] = jdata[d]

#print(category_map)
print(server_category_map)
if category_map == server_category_map:
    print('category maps match!')
else:
    print('local data set and server category maps do NOT match!')
    #raise

###############################################################################
# Evaluate the Model
###############################################################################

summary_fail = 0
summary_success = 0

count = 0
for item in network_input_index:
    #print(item)
    metadata = network_input_index[item]
    print("(%s%s)(%s)" % (config.DATA_DIRECTORY, item, metadata[1]))

    filename = "%s%s" % (config.DATA_DIRECTORY, item)
    with open(filename, 'rb') as file:
        file_content = file.read()

    base64_encoded_content = file_content.encode('base64')

    outjson = {}
    outjson['data'] = base64_encoded_content

    json_data = json.dumps(outjson)

    prediction = {}
    category = {}

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
                if 'prediction' in response.json():
                    prediction = response.json()['prediction']
                    category = server_category_map[str(prediction)]
                    #print("server prediction: (%s)" % prediction)
    except:
        print('.')
        #raise

    #print prediction
    # index = 0
    # keyed_prediction = {}
    # for value in prediction:
    #     #print("%i, %s" % (index, value))
    #     keyed_prediction[index] = value
    #     index += 1
    # #print(keyed_prediction)
    # max_item = max(keyed_prediction.iteritems(), key=operator.itemgetter(1))
    # #print(max_item)
    #
    # raw_index = int(max_item[0])
    # raw_value = float(max_item[1])
    # #print("(%i)(%f)" % (raw_index, raw_value))
    #
    # for key, value in server_category_map.iteritems():
    #     if value == raw_index:
    #         #print(key)
    #         readable_category = key
    #         readable_index = value
    #
    # print(readable_category)

    # Get the local image index for comparision against the server prediction
    # print(metadata[1])
    # print(server_category_map[metadata[1]])
    #print("%s" % (metadata[1]))
    #print("server prediction: (%s)" % category)

    if category == metadata[1]:
    #if readable_category == metadata[1]:
     print('correct!')
     summary_success += 1
    else:
     print('FAIL')
     summary_fail += 1

    if count > 1000:
     count += 1
     break
    else:
     count += 1

    #for key, value in sorted(prediction.iteritems(), key=lambda (k, v): (v, k)):
    #    #print "%s: %s" % (key, value)
    #    cv2.putText(camera_capture, "%s: %s" % (key, value), (5, 20 + 20 * i), font, 0.5, (255, 255, 255), 1)
    #    i -= 1
    #    if i < 0:
    #        i = 4

success_percentage = (float(summary_success) / count) * 100
failure_percentage = (float(summary_fail) / count) * 100

print('print our summary')
print("success: (%i)" % summary_success)
print("failures: (%i)" % summary_fail)
print("total tests: (%i)" % count)
print("success rate: (%f)" % success_percentage)
print("failure rate: (%f)" % failure_percentage)
