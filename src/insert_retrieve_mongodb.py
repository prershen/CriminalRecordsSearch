#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:22:56 2019

@author: pratz
"""
"""database-mugshot
   collection-photos
   attr-_id,img"""

from pymongo import MongoClient
from bson import ObjectId
import gridfs
import os,shutil
from PIL import Image
import base64
import codecs
import face_recognition
import pprint

client=MongoClient();
db=client.mugshot;   

def insert(__id,path,offence_id):
    with open(path, "rb") as imageFile:
      string = base64.b64encode(imageFile.read())   
    db.photos.insert_one({"_id":__id,"img":string,"offence_id":offence_id})
    
"""insert("P_1","/home/pratz/test_photos/meghna.jpeg",["D_1","V_1"])"""

def retrieve(__id):    
  img_dict=db.photos.find_one({"_id":__id},{"img":1,"offence_id":1,"_id":0})
  string=img_dict["img"] 
  offence_id=img_dict["offence_id"]
  fh = open("/home/pratz/retrieved_photos/{}.jpg".format(__id), "wb")
  fh.write(codecs.decode(string,'base64'))
  fh.close()
  return offence_id

"""retrieve("P_1")"""

def face_match(known_img_path,unknown_img_path):
  known_image = face_recognition.load_image_file(known_img_path)
  unknown_image = face_recognition.load_image_file(unknown_img_path)
  biden_encoding = face_recognition.face_encodings(known_image)[0]
  unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
  results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
  return results
  
def face_search(unknown_img_path):
  for itm in db.photos.find({}):
      retrieve(itm.get('_id'))
  for filename in os.listdir("/home/pratz/retrieved_photos"):
    res=face_match("/home/pratz/retrieved_photos/"+filename,unknown_img_path)
    if res==True:
       img = Image.open("/home/pratz/retrieved_photos/"+filename)
       img = img.resize((1024,768))
       l=filename.split('.')
       __id=l[0]
       break
    else:
      __id=None        
  return __id  

"""_id=face_search("/home/pratz/test_photos/meghna.jpeg")
print(_id)"""
