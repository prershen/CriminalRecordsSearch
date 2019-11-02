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

def insert(__id,path):
    with open(path, "rb") as imageFile:
      str = base64.b64encode(imageFile.read())   
    db.photos.insert_one({"_id":__id,"img":str})

def retrieve(__id):    
  img_dict=db.photos.find_one({"_id":__id})
  for k in img_dict.keys():
     str=img_dict[k];
  fh = open("/home/pratz/retrieved_photos/{}.jpg".format(__id), "wb")
  fh.write(codecs.decode(str,'base64'))
  fh.close()

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
    if res==False:
      __id=None
        
  folder = "/home/pratz/retrieved_photos"      
  for the_file in os.listdir(folder):
         file_path = os.path.join(folder, the_file)
         try:
           if os.path.isfile(file_path):
            os.unlink(file_path)
           elif os.path.isdir(file_path): shutil.rmtree(file_path)
         except Exception as e:
            print(e)
  return __id
