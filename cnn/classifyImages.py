# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
# import numpy as np
# from keras.applications.vgg16 import decode_predictions
import json
import os

vggClasses = dict()
counter = 0
with open('allVgg.txt','r') as vf:
    for l in vf:
        split = l.split(': ')
        if split[1].strip() == 'food':
            counter+=1
print(1000 / counter)

weigthCorection = [47.6,17.54,8.13,24.39,22.22,8.47,30.3,2.38,17.24,27.77,20.83]
# print(0.90 * 2.38)
# print(18.75 / 20)

# model = VGG16(weights='imagenet', include_top=True)

# rootdir = '../../books/2018'

# for subdir in os.walk(rootdir):
#     path = subdir[0]
#     if len(path) > 13:
#         img_path = path+'/cover.jpeg'
#         print(img_path)
#         imCl = dict()
#         imCl["vggClass0"] = 0
#         imCl["vggClass1"] = 0
#         imCl["vggClass2"] = 0
#         imCl["vggClass3"] = 0
#         imCl["vggClass4"] = 0
#         imCl["vggClass5"] = 0
#         imCl["vggClass6"] = 0
#         imCl["vggClass7"] = 0
#         imCl["vggClass8"] = 0
#         imCl["vggClass9"] = 0
#         imCl["vggClass10"] = 0
#         try:
#             img = image.load_img(img_path, target_size=(224, 224))
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = preprocess_input(x)
#             features = model.predict(x)
#             i = 0
#             sc1 = 0
#             sc2 = 0
#             sc3 = 0
#             sc4 = 0
#             sc5 = 0
#             sc6 = 0
#             sc7 = 0
#             sc8 = 0
#             sc9 = 0
#             sc10 = 0
#             sc11 = 0


#             for f in features[0]:
#                 if i != 917 and i != 916 and i != 921 and i != 922 and i!= 549 and i != 446 and i!= 692:
#                     if vggClasses[str(i)] == 'item':
#                         sc8 += f
#                     elif vggClasses[str(i)] == 'bird':
#                         sc2 += f   
#                     elif vggClasses[str(i)] == 'animal':
#                         sc3 += f 
#                     elif vggClasses[str(i)] == 'bug':
#                         sc4 += f 
#                     elif vggClasses[str(i)] == 'reptile':
#                         sc5 += f  
#                     elif vggClasses[str(i)] == 'dog':
#                         sc6 += f  
#                     elif vggClasses[str(i)] == 'clothes':
#                         sc7 += f  
#                     elif vggClasses[str(i)] == 'fish':
#                         sc1 += f 
#                     elif vggClasses[str(i)] == 'vehicle':
#                         sc9 += f  
#                     elif vggClasses[str(i)] == 'building':
#                         sc10 += f  
#                     else:
#                         sc11 += f  
#                 i+=1


#             imCl["vggClass0"] = round(sc1,5)
#             imCl["vggClass1"] = round(sc2,5)
#             imCl["vggClass2"] = round(sc3,5)
#             imCl["vggClass3"] = round(sc4,5)
#             imCl["vggClass4"] = round(sc5,5)
#             imCl["vggClass5"] = round(sc6,5)
#             imCl["vggClass6"] = round(sc7,5)
#             imCl["vggClass7"] = round(sc8,5)
#             imCl["vggClass8"] = round(sc9,5)
#             imCl["vggClass9"] = round(sc10,5)
#             imCl["vggClass10"] = round(sc11,5)
            
#         except:
#             print('Failed')


#         with open(path+'/vgg16Classification.json', 'w') as fp:
#             json.dump(imCl, fp, sort_keys=False, indent=4)
    

