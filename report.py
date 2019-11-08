import mysql.connector
from mysql.connector import Error
from tkinter import *
from datetime import *
from pymongo import MongoClient
from bson import ObjectId
import gridfs
import os,shutil
from PIL import Image
import base64
import codecs
import face_recognition
import pprint

class report:
    connection = mysql.connector.connect(host='localhost',
                                         database='dbmslab',
                                         user='root',
                                         password='Perushenoy@99')
    client=MongoClient();
    db=client.mugshot;  
    def __init__(self, p_id):
        self.p_id=p_id
        self.mycursor=self.connection.cursor()
        sql = "SELECT * FROM personal_details WHERE id like '"+p_id+"';"
        print(sql)
        self.mycursor.execute(sql)

        self.myresult = self.mycursor.fetchone()
            #print ("Criminal_Id: %s\nName: %s\nDOB: %s\nGender: %s\nAge: %d\nAddress: %s\nEye_color: %s\nHair_color: %s\n Height: %s\nWeight: %s\n"%(x["id"],x["criminal_name"],x["DOB"],x["gender"],x["age"],x["address"],x["eye_color"],x["hair"],x["height"],x["weight"]))
            #print ("Criminal_Id: %s\nName: %s\nDOB: %s\nGender: %s\nAge: %d\nAddress: %s\nEye_color: %s\nHair_color: %s\n Height: %s\nWeight: %s\n"%(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]))
        self.tlist=[] #traffic
        self.rlist=[] #robbery
        self.vlist=[] #violence
        self.dlist=[] #drug
        offence_id=retrieve(p_id)
        """put mongodb statement to get all offence ids of criminal in tlist,rlist,vlist and dlist for p_id
        if len(tlist)!=0:
            sql="SELECT * FROM traffic WHERE offence_id IN("
            flag=0
            for i in self.tlist:
                if flag==0:
                    sql=sql+"'"+i"'"
                flag=flag+1
                sql=sql+",'"+i"'"
            sql=sql+");
            self.mycursor.execute(sql)
            self.traffic=self.mycursor.fetchAll()
        
        if len(rlist)!=0:
            sql="SELECT * FROM robbery WHERE offence_id IN("
            flag=0
            for i in self.rlist:
                if flag==0:
                    sql=sql+"'"+i"'"
                flag=flag+1
                sql=sql+",'"+i"'"
            sql=sql+");
            self.mycursor.execute(sql)
            self.robbery=self.mycursor.fetchAll()

        if len(vlist)!=0:
            sql="SELECT * FROM violence WHERE offence_id IN("
            flag=0
            for i in self.vlist:
                if flag==0:
                    sql=sql+"'"+i"'"
                flag=flag+1
                sql=sql+",'"+i"'"
            sql=sql+");
            self.mycursor.execute(sql)
            self.violence=self.mycursor.fetchAll()

        if len(dlist)!=0:
            sql="SELECT * FROM drug WHERE offence_id IN("
            flag=0
            for i in self.dlist:
                if flag==0:
                    sql=sql+"'"+i"'"
                flag=flag+1
                sql=sql+",'"+i"'"       
            sql=sql+");"
            self.mycursor.execute(sql)
            self.drug=self.mycursor.fetchAll()
        
        """

        self.root = Tk() 
        self.root.title('Report of found criminal')
        Button(self.root,text="Personal Details",command=self.personal).grid(row=1,column=1,padx=10,pady=10)
        #b1.pack()
        Button(self.root,text="Criminal History",command=self.offence).grid(row=2,column=1,padx=10,pady=10)
        #b2.pack()
        self.root.mainloop()

    def personal(self):
        self.root1=Tk()
        self.root1.title('Personal Details')
        l1=Label(self.root1,text="Criminal Id: "+self.myresult[0])
        l1.grid(row=1,column=1,padx=10,pady=10)
        l2=Label(self.root1,text="Name: "+self.myresult[1])
        l2.grid(row=2,column=1,padx=10,pady=10)
        l3=Label(self.root1,text="DOB: "+self.myresult[2].strftime('%Y-%m-%d'))
        l3.grid(row=3,column=1,padx=10,pady=10)
        l4=Label(self.root1,text="Gender: "+self.myresult[3])
        l4.grid(row=4,column=1,padx=10,pady=10)
        l5=Label(self.root1,text="Age: "+str(self.myresult[4]))
        l5.grid(row=5,column=1,padx=10,pady=10)
        l6=Label(self.root1,text="Address: "+self.myresult[5])
        l6.grid(row=6,column=1,padx=10,pady=10)
        l7=Label(self.root1,text="Eye_color: "+self.myresult[6])
        l7.grid(row=7,column=1,padx=10,pady=10)
        l8=Label(self.root1,text="Hair_color: "+self.myresult[7])
        l8.grid(row=8,column=1,padx=10,pady=10)
        l9=Label(self.root1,text="Height: "+self.myresult[8])
        l9.grid(row=9,column=1,padx=10,pady=10)
        l10=Label(self.root1,text="Weight: "+self.myresult[9])
        l10.grid(row=10,column=1,padx=10,pady=10)
        
        self.root1.mainloop()
    def offence(self):
        self.root2=Tk()
        self.root2.title('Criminal History')
        self.root2.mainloop()
    
    
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
found=report("P_1")
