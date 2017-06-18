import base64
import cv, cv2
import sys
import json

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)

#print 'image: ', str(sys.argv[1])
image_to_convert = str(sys.argv[1])

with open(image_to_convert, 'rb') as file:
    file_content = file.read()

base64_encoded_content = file_content.encode('base64')

outjson = {}
outjson['data'] = base64_encoded_content

json_data = json.dumps(outjson)
print json_data
