# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:12:31 2023

@author: COMPUTER
"""



import tkinter as tk
import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.naive_bayes import GaussianNB
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

root = tk.Tk()
root.title("STRESS BOT")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('log.jpg')

image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="STRESS BOT", font=('times', 35,' bold '), height=1, width=55,bg="violet Red",fg="Black")
lbl.place(x=0, y=10)


le = LabelEncoder()






def Model_Training():
    dataset = pd.read_csv('Book1.csv')
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

   
    
    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    X = scalerX.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=0)
    
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear',random_state=0)
    classifier.fit(XTrain,yTrain)
    yPred = classifier.predict(XTest)
    mse = mean_squared_error(yTest,yPred)
    r = r2_score(yTest,yPred)
    mae = mean_absolute_error(yTest,yPred)
    accuracy = accuracy_score(yTest,yPred)
    repo = (classification_report(yTest, yPred))
    print(repo)
    print("Support Vector Machine :")
    #print("Accuracy = ", accuracy*100)
    from joblib import dump
    dump (classifier,"SVM1.joblib")
    print("Model saved as SVM1.joblib")
    
def call_file():
     
    from subprocess import call
    call(["python","stress.py"])



def window():
    root.destroy()



# button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
#                     text="TRAIN ", command=Model_Training, width=15, height=2)
# button3.place(x=5, y=100)



button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="TEST", command=call_file, width=15, height=2)
button4.place(x=5, y=200)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=300)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''