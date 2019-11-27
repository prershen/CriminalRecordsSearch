connection = mysql.connector.connect(host='localhost',
                                         database='dbms',
                                         user='admin',
                                         password='root1234')
        self.mycursor=self.connection.cursor()
        Label(self.root, text="Please enter details below to login").pack()
        Label(self.root, text="").pack()
    
        global username_verify
        global password_verify
    
        username_verify = StringVar()
        password_verify = StringVar()
    
    
        Label(self.root, text="Username * ").pack()
        username_login_entry = Entry(self.root, textvariable=username_verify)
        username_login_entry.pack()
        Label(self.root, text="").pack()
        Label(self.root, text="Password * ").pack()
        password__login_entry = Entry(self.root, textvariable=password_verify, show= '*')
        password__login_entry.pack()
        Label(self.root, text="").pack()
        Button(self.root, text="Login", width=10, height=1, command=self.login_verification).pack()

    def login_verification(self):
        sql = "SELECT * FROM login WHERE user like '"+username_verify+"' and  password like '"+password_verify+"';"
        self.mycursor.execute(sql)
        self.login_result = self.mycursor.fetchone()
        if self.login_result!=None:
            self.preview2=Toplevel(self.root)
            ttk.Button(self.root, text='Upload a sketch', command=self.get_image).pack()
        else:
            self.preview2=Toplevel(self.root)
            ttk.Label(self.preview2,text="Wrong credentials. Try Again").pack()
