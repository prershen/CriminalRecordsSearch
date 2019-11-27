#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:22:14 2019

@author: pratz
"""

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image


from pymongo import MongoClient
from bson import ObjectId

import os
from PIL import Image
import mysql.connector
from mysql.connector import Error
from tkinter import *
from datetime import *

from bson import ObjectId
import tkinter
from tkinter import ttk
from ttkthemes import themed_tk as tk
import ttk
import base64
import codecs
import face_recognition
import pprint

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.models import load_model

from cv2 import cv2
import keras.backend as K
from PIL import Image, ImageFilter
import PIL

from ttkthemes import themed_tk as tk
from tkinter import ttk

client=MongoClient()
db=client.mugshot

class Uploader(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        
        self.root = Canvas(height=100,width=100)
        self.root.grid()
        self.title('Criminal Search')  
        
        #self.logo = ImageTk.PhotoImage(Image.open("/home/pratz/Criminal-Check.png"))

        #Label(self.root, image=self.logo).grid(row=0,column=0)
        
        #Button(self.root, text='UPLOAD A SKETCH',command=self.get_image,bg="red4",fg="white",height=3,width=20).grid(padx=10,pady=10)
        
        Label(self.root, text="Please enter details below to login").grid(padx=10,pady=10)
       # Label(self.root, text="").grid(padx=10,pady=10)
        self.login = ImageTk.PhotoImage(Image.open("/home/pratz/login1.jpeg"))

        #Label(self.root, image=self.login).grid(row=0,column=0)
        background=Label(self.root, image=self.login).place(x=0,y=0,relwidth=1, relheight=1)
        global username_verify
        global password_verify
    
        username_verify = StringVar()
        password_verify = StringVar()
    
        
        Label(self.root, text="USERNAME",font='gothic 16 bold',fg="white",bg="Dodgerblue2").grid(padx=170,pady=10)
        username_login_entry = Entry(self.root, textvariable=username_verify,width=40)
        username_login_entry.grid(padx=10,pady=10,ipady=7)
        #Label(self.root, text="").grid(padx=100,pady=100)
        Label(self.root, text="PASSWORD",font='gothic 16 bold',fg="white",bg="Dodgerblue2").grid(padx=10,pady=10)
        password__login_entry = Entry(self.root, textvariable=password_verify, show= '*',width=40)
        password__login_entry.grid(padx=170,pady=10,ipady=7)
        #Label(self.root, text="").grid(padx=10,pady=10)
        Button(self.root, text="LOGIN", width=20, height=2, command=self.login_verification,bg="red2",fg="white").grid(padx=10,pady=70)

    def login_verification(self):
        
        self.connection = mysql.connector.connect(host='localhost',
                                         database='dbms',
                                         user='admin',
                                         password='root1234')
        self.mycursor=self.connection.cursor()
        
        sql = "SELECT * FROM login WHERE user like '"+str(username_verify.get())+"' and  password like '"+str(password_verify.get())+"';"
        self.mycursor.execute(sql)
        self.login_result = self.mycursor.fetchone()
        if self.login_result!=None:
            self.preview2=Toplevel(self.root)
            self.logo = ImageTk.PhotoImage(Image.open("/home/pratz/Criminal-Check.png"))
            Label(self.preview2, image=self.logo).grid(row=0,column=0)
            Button(self.preview2, text='UPLOAD A SKETCH',command=self.get_image,bg="red4",fg="white",height=2,width=20,font='TIMES 18 bold').grid(padx=10,pady=10)
        else:
            self.preview2=Toplevel(self.root)
            Label(self.preview2,text="Wrong credentials. Try Again").grid(padx=10,pady=10)


    def get_image(self):
        
        self.file_name = askopenfilename(filetypes=[('JPEG FILES', '*.jpg')])
        
        self.image = ImageTk.PhotoImage(Image.open(self.file_name))
        self.preview = Toplevel(bg='azure')
        
        self.sketch=Label(self.preview,image=self.image)
        self.sketch.grid(row=0, column=0, padx=100, pady=25)
        self.photo= Button(self.preview, command=self.createPhoto, text="CREATE PHOTO",bg="black",fg="white",height=3,width=20)
        self.photo.grid(row=1,padx=10, pady=5)
        
    def predict_photo(self):
        json_file = open('/home/pratz/Downloads/dataset/results/model28000.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/home/pratz/Downloads/dataset/results/model28000.h5")
        print("Loaded model from disk")
        im = cv2.imread(self.file_name,0)
        plt.imshow(im)
        plt.show()    
        im = cv2.resize(im,(64,64))
        im = im.reshape((1,4096))
        img = loaded_model.predict(im)[0]
        #img=cv2.GaussianBlur(img, (3,3), 0)
        #img = cv2.addWeighted(blur,1.5,img,-0.5,0)
        gen_img = (1/2.5) * img + 0.5 
        #cv2.imwrite("/home/pratz/predicted_photo.jpg",gen_img)
        self.predicted_filename="/home/pratz/predicted_photo.jpg"
        
        plt.imsave(self.predicted_filename,gen_img)       
        plt.imshow(gen_img)
        plt.show()
        

    def createPhoto(self):
        self.preview1=Toplevel(bg="ivory2")
        self.predict_photo()
        self.image = ImageTk.PhotoImage(Image.open(self.predicted_filename))
        
        self.photo=Label(self.preview1,image=self.image)
        self.photo.grid(row=0,column=1,padx=100,pady=25)
        self.search=Button(self.preview1,command=self.searching,text="SEARCH",bg="black",fg="white",height=3,width=20)
        self.search.grid(row=1,column=1,padx=10,pady=10)
        
        
    def searching(self):
        #calls the processing      
        
        print(self.predicted_filename)
        self.pid=face_search(self.predicted_filename)
        if(self.pid==None):             
             self.notfound=Label(self.preview1,text="CRIMINAL RECORD NOT FOUND!",fg="red4",font='TIMES 18 bold').grid(row=3,column=1,padx=10,pady=10)
        if(self.pid!=None): 
             found=report(self.pid)
             
       
        


class report:
    connection = mysql.connector.connect(host='localhost',
                                         database='dbms',
                                         user='admin',
                                         password='root1234')
    client=MongoClient();
    db=client.mugshot;  
    def __init__(self, p_id):
        self.p_id=p_id
        self.mycursor=self.connection.cursor()
        sql = "SELECT * FROM personal_details WHERE id like '"+self.p_id+"';"
        print(sql)
        self.root = Tk()      
        
        self.root.title('Report of found criminal')
        self.mycursor.execute(sql)
        self.myresult = self.mycursor.fetchone()
        
        if self.mycursor.rowcount>0:
                #print ("Criminal_Id: %s\nName: %s\nDOB: %s\nGender: %s\nAge: %d\nAddress: %s\nEye_color: %s\nHair_color: %s\n Height: %s\nWeight: %s\n"%(x["id"],x["criminal_name"],x["DOB"],x["gender"],x["age"],x["address"],x["eye_color"],x["hair"],x["height"],x["weight"]))
                #print ("Criminal_Id: %s\nName: %s\nDOB: %s\nGender: %s\nAge: %d\nAddress: %s\nEye_color: %s\nHair_color: %s\n Height: %s\nWeight: %s\n"%(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]))
            Button(self.root,text="PERSONAL DETAILS",command=self.personal,bg="black",fg="white",height=3,width=20).grid(row=1,column=1,padx=100,pady=30)
            #b1.pack()
            Button(self.root,text="CRIMINAL HISTORY",command=self.offence,bg="black",fg="white",height=3,width=20).grid(row=2,column=1,padx=100,pady=50)
            #b2.pack()
            self.tlist=[] #traffic
            self.rlist=[] #robbery
            self.vlist=[] #violence
            self.dlist=[] #drug
            offence_id=retrieve(p_id)
            print(offence_id)
            for id in offence_id:
                if(id.startswith('T')):
                    self.tlist.append(id)
                elif id.startswith('R'):
                    self.rlist.append(id)
                elif id.startswith('V'):
                    self.vlist.append(id)
                elif id.startswith('D'):
                    self.dlist.append(id)
            if len(self.tlist)!=0:
                sql="SELECT * FROM traffic WHERE offence_id IN("
                flag=0
                for i in self.tlist:
                    if flag==0:
                        sql=sql+"'"+i+"'"
                    flag=flag+1
                    sql=sql+",'"+i+"'"
                sql=sql+");"
                self.mycursor.execute(sql)
                self.traffic=self.mycursor.fetchall()
            
            if len(self.rlist)!=0:
                sql="SELECT * FROM robbery WHERE offence_id IN("
                flag=0
                for i in self.rlist:
                    if flag==0:
                        sql=sql+"'"+i+"'"
                    flag=flag+1
                    sql=sql+",'"+i+"'"
                sql=sql+");"
                self.mycursor.execute(sql)
                self.robbery=self.mycursor.fetchall()
            if len(self.vlist)!=0:
                sql="SELECT * FROM violence WHERE offence_id IN("
                flag=0
                for i in self.vlist:
                    if flag==0:
                        sql=sql+"'"+i+"'"
                    flag=flag+1
                    sql=sql+",'"+i+"'"
                sql=sql+");"
                self.mycursor.execute(sql)
                self.violence=self.mycursor.fetchall()
            if len(self.dlist)!=0:
                sql="SELECT * FROM drug WHERE offence_id IN("
                flag=0
                for i in self.dlist:
                    if flag==0:
                        sql=sql+"'"+i+"'"
                    flag=flag+1
                    sql=sql+",'"+i+"'"       
                sql=sql+");"
                self.mycursor.execute(sql)
                self.drug=self.mycursor.fetchall()
        
            
            self.root.mainloop()
        else:
            Label(self.root,text="Criminal Record not found!").grid(row=1,column=1,padx=10,pady=10)
            print("Not found")

    def personal(self):
        self.root1 = Tk()
        
        
        self.root1.title('Personal Details')
        l1=Label(self.root1,text="Criminal Id: "+self.myresult[0])
        l1.grid(row=1,column=1,padx=30,pady=10)
        l2=Label(self.root1,text="Name: "+self.myresult[1])
        l2.grid(row=2,column=1,padx=30,pady=10)
        l3=Label(self.root1,text="DOB: "+self.myresult[2].strftime('%Y-%m-%d'))
        l3.grid(row=3,column=1,padx=30,pady=10)
        l4=Label(self.root1,text="Gender: "+self.myresult[3])
        l4.grid(row=4,column=1,padx=30,pady=10)
        l5=Label(self.root1,text="Age: "+str(self.myresult[4]))
        l5.grid(row=5,column=1,padx=30,pady=10)
        l6=Label(self.root1,text="Address: "+self.myresult[5])
        l6.grid(row=6,column=1,padx=30,pady=10)
        l7=Label(self.root1,text="Eye_color: "+self.myresult[6])
        l7.grid(row=7,column=1,padx=30,pady=10)
        l8=Label(self.root1,text="Hair_color: "+self.myresult[7])
        l8.grid(row=8,column=1,padx=30,pady=10)
        l9=Label(self.root1,text="Height: "+self.myresult[8])
        l9.grid(row=9,column=1,padx=30,pady=10)
        l10=Label(self.root1,text="Weight: "+self.myresult[9])
        l10.grid(row=10,column=1,padx=30,pady=10)
        
        self.root1.mainloop()
    def offence(self):
        self.root2 = Tk()
        
        
        self.root2.title('Criminal History')
        #Traffic       
        i=j=k=l=0;
        
        for i in range(0,len(self.tlist)):
            Label(self.root2,text="Offence_Id: "+self.traffic[i][0]).grid(row=i+1,column=1,padx=10,pady=10)
            Label(self.root2,text="Date of offence: "+self.traffic[i][1].strftime('%Y-%m-%d')).grid(row=i+2,column=1,padx=10,pady=10)
            Label(self.root2,text="type: "+self.traffic[i][2]).grid(row=i+3,column=1,padx=10,pady=10)
            Label(self.root2,text="status: "+self.traffic[i][3]).grid(row=i+4,column=1,padx=10,pady=10)
            i=i+4
            print(len(self.tlist))
            
        #Robbery
        for j in range(0,len(self.rlist)):
            Label(self.root2,text="Offence_Id: "+self.robbery[j][0]).grid(row=i+j+1,column=1,padx=10,pady=10)
            Label(self.root2,text="Date of offence: "+self.robbery[j][1].strftime('%Y-%m-%d')).grid(row=i+j+2,column=1,padx=10,pady=10)
            Label(self.root2,text="Items: "+self.robbery[j][2]).grid(row=i+j+3,column=1,padx=10,pady=10)
            Label(self.root2,text="Status: "+self.robbery[j][3]).grid(row=i+j+4,column=1,padx=10,pady=10)
            Label(self.root2,text="Case Details: "+self.robbery[j][4]).grid(row=i+j+5,column=1,padx=10,pady=10)
            i=i+5
            print(len(self.rlist))
            
        #Violence
        for k in range(0,len(self.vlist)):
            Label(self.root2,text="Offence_Id: "+self.violence[k][0]).grid(row=i+j+k+1,column=1,padx=10,pady=10)
            Label(self.root2,text="Date of offence: "+self.violence[k][1].strftime('%Y-%m-%d')).grid(row=i+j+k+2,column=1,padx=10,pady=10)
            Label(self.root2,text="type: "+self.violence[k][2]).grid(row=i+j+k+3,column=1,padx=10,pady=10)
            Label(self.root2,text="status: "+self.violence[k][3]).grid(row=i+j+k+4,column=1,padx=10,pady=10)
            Label(self.root2,text="Case Details: "+self.violence[k][4]).grid(row=i+j+k+5,column=1,padx=10,pady=10)
            i=i+5
            print(len(self.vlist))
            
        #Drug
        for l in range(0,len(self.dlist)):
            Label(self.root2,text="Offence_Id: "+self.drug[l][0]).grid(row=i+j+k+l+1,column=1,padx=10,pady=10)
            Label(self.root2,text="Date of offence: "+self.drug[l][1].strftime('%Y-%m-%d')).grid(row=i+j+k+l+2,column=1,padx=10,pady=10)
            Label(self.root2,text="type: "+self.drug[l][2]).grid(row=i+j+k+l+3,column=1,padx=10,pady=10)
            Label(self.root2,text="status: "+self.drug[l][3]).grid(row=i+j+k+l+4,column=1,padx=10,pady=10)
            Label(self.root2,text="Case Details: "+self.drug[l][4]).grid(row=i+j+k+l+5,column=1,padx=10,pady=10)
            i=i+5
            print(len(self.dlist))
            
        self.root2.mainloop()
    
    
    def insert(self,__id,path,offence_id):
      with open(path, "rb") as imageFile:
        string = base64.b64encode(imageFile.read())   
      db.photos.insert_one({"_id":__id,"img":string,"offence_id":offence_id})
    
   
def retrieve(__id):    
      img_dict=db.photos.find_one({"_id":__id},{"img":1,"offence_id":1,"_id":0})
      string=img_dict["img"] 
      offence_id=img_dict["offence_id"]
      fh = open("/home/pratz/retrieved_photos/{}.jpg".format(__id), "wb")
      fh.write(codecs.decode(string,'base64'))
      fh.close()
      return offence_id

def face_match(known_img_path,unknown_img_path):
      known_image = face_recognition.load_image_file(known_img_path)
      unknown_image = face_recognition.load_image_file(unknown_img_path)
      biden_encoding = face_recognition.face_encodings(known_image)[0]
      unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
      results = face_recognition.compare_faces([biden_encoding], unknown_encoding,tolerance=0.7)
      return results[0]
  
def face_search(unknown_img_path):
         
          for itm in db.photos.find({}):
              retrieve(itm.get('_id'))
          for filename in os.listdir("/home/pratz/retrieved_photos"):
            res=face_match("/home/pratz/retrieved_photos/"+filename,unknown_img_path)
            if res==True:
               print("Face search done")
               img = Image.open("/home/pratz/retrieved_photos/"+filename)
               img = img.resize((1024,768))
               l=filename.split('.')
               __id=l[0]
               break
            else:
                print("Facesearch not done")  
                __id=None        
          return __id

"""_id=face_search("/home/pratz/test_photos/meghna.jpeg")
print(_id)"""
app = Uploader()
app.mainloop()
