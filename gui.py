from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

from pymongo import MongoClient
from bson import ObjectId
import gridfs
import os
from PIL import Image
#client=MongoClient()
#db=client.dbmslab

class Uploader(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.root = Canvas()
        self.root.grid()
        self.title('Criminal Search')  
        Button(self.root, text='Upload an image', command=self.get_image).grid(padx=50,pady=5)
        Label(text='Click preview picture to upload').grid(pady=5)

    def get_image(self):
        self.file_name = askopenfilename(filetypes=[('JPEG FILES', '*.jpg')])

        self.image = ImageTk.PhotoImage(Image.open(self.file_name))
        self.preview = Toplevel()
        self.sketch=Label(self.preview,image=self.image)
        self.sketch.grid(row=0, column=1, padx=10, pady=10)
        self.save = Button(self.preview, command=self.saving, text="Save")
        self.save.grid(row=1, column=1, padx=10, pady=10)
        self.photo= Button(self.preview, command=self.createPhoto, text="Create Photo")
        self.photo.grid(row=3, column=1, padx=10, pady=10)

    def saving(self):
        #gfsPhoto =  gridfs.GridFS(db, "sketch")
        #img=open(self.file_name,'rb')
        #gfsFile = gfsPhoto.put(img,_id='S_1')
        print("Saving")
    def createPhoto(self):
        self.preview1=Toplevel()
        self.photo=Label(self.preview1)#,image=self.photo)
        self.photo.grid(row=0,column=1,padx=10,pady=10)
        self.search=Button(self.preview1,command=self.searching,text="Search")
        self.search.grid(row=1,column=1,padx=10,pady=10)
    def searching(self):
        #calls the processing
        Label(self.preview1,text="The record is found!")

app = Uploader()
app.mainloop()
