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
import os
from PIL import Image
import base64
import codecs

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
  fh = open("/home/pratz/{}.jpg".format(__id), "wb")
  fh.write(codecs.decode(str,'base64'))
  fh.close()

insert("P_1","test.jpg")
retrieve("P_1")