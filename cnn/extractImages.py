import os
import json
import requests

rootdir = '../books'
processed = 0
for subdir in os.walk(rootdir):
    path = subdir[0]
    if(len(path) > 13):
        with open(path+'/details.json')as f:
            data = json.load(f)
            
            bookImageFilename = path + '/cover.jpeg'
            print(processed)
            f = open(bookImageFilename,'wb')
            f.write(requests.get(data['imgUrl']).content)
            f.close()
            processed = processed + 1
