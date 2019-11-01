import mysql.connector
from mysql.connector import Error
from tkinter import *

class report:
    connection = mysql.connector.connect(host='localhost',
                                         database='dbmslab',
                                         user='root',
                                         password='Perushenoy@99')
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
        root1=Tk()
        root1.title('Personal Details')
        l1=Label(self.root,text="Criminal Id: "+self.myresult[0])
        l1.pack()
        l2=Label(self.root,text="Name: "+self.myresult[1])
        l2.pack()
        l3=Label(self.root,text="DOB: "+self.myresult[2])
        l3.pack()
        l4=Label(self.root,text="Gender: "+self.myresult[3])
        l4.pack()
        l5=Label(self.root,text="Age: "+self.myresult[4])
        l5.pack()
        l6=Label(self.root,text="Address: "+self.myresult[5])
        l6.pack()
        l7=Label(self.root,text="Eye_color: "+self.myresult[6])
        l7.pack()
        l8=Label(self.root,text="Hair_color: "+self.myresult[7])
        l8.pack()
        l9=Label(self.root,text="Height: "+self.myresult[8])
        l9.pack()
        l10=Label(self.root,text="Weight: "+self.myresult[9])
        l10.pack()
        root1.mainloop()
    def offence(self):
        root2=Tk()
        root2.title('Criminal History')

        root2.mainloop()
found=report("P_1")
