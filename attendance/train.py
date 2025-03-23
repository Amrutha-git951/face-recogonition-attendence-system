import tkinter as tk
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time

window = tk.Tk()
window.title("Face_Recogniser")
window.geometry('1280x720')
window.configure(background='#2C3E50')  # Set a dark background color

message = tk.Label(window, text="Attendance-Monitoring using face recognition", bg="#3498DB", fg="white",
                   width=45, height=3, font=('times', 30, 'italic bold underline'))
message.place(x=150, y=20)

labels_color = "#ECF0F1"  # Light gray
entry_bg_color = "#34495E"  # Dark gray
entry_fg_color = "white"
button_bg_color = "#3498DB"  # Blue

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="white", bg="#3498DB",
               font=('times', 15, 'bold'))
lbl.place(x=350, y=200)

txt = tk.Entry(window, width=20, bg=entry_bg_color, fg=entry_fg_color, font=('times', 15, 'bold'))
txt.place(x=650, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="white", bg="#3498DB", height=2,
                font=('times', 15, 'bold'))
lbl2.place(x=350, y=275)

txt2 = tk.Entry(window, width=20, bg=entry_bg_color, fg=entry_fg_color, font=('times', 15, 'bold'))
txt2.place(x=650, y=290)

lbl3 = tk.Label(window, text="Notification:", width=20, fg="white", bg="#3498DB", height=2,
                font=('times', 15, 'bold underline'))
lbl3.place(x=350, y=375)

message = tk.Label(window, text="", bg="#3498DB", fg="white", width=30, height=2,
                   font=('times', 15, 'bold'))
message.place(x=650, y=375)

lbl3 = tk.Label(window, text="Attendance:", width=20, fg="white", bg="#3498DB", height=2,
                font=('times', 15, 'bold underline'))
lbl3.place(x=350, y=450)

message2 = tk.Label(window, text="", fg="white", bg="#3498DB", width=30, height=2,
                    font=('times', 15, 'bold'))
message2.place(x=650, y=450)

def clear():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')
    message.configure(text="")

def is_alphanumeric(s):
    return all(c.isalpha() or c.isspace() for c in s)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def capture_images(Id, name):
    cam = cv2.VideoCapture(0)
    # Add a brief delay to allow the camera to stabilize
    time.sleep(1)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while sampleNum < 60:  # Capture 60 samples
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Incrementing sample number 
            sampleNum += 1
            # Saving the captured face in the dataset folder TrainingImage
            cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
            # Display the frame
            cv2.imshow('frame', img)
        # Wait for 100 milliseconds 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return sampleNum

def TakeImages():
    Id = txt.get()
    name = txt2.get()
    if is_number(Id) and is_alphanumeric(name):
        num_samples = capture_images(Id, name)
        res = f"Images Saved for ID: {Id}, Name: {name}, Total Samples Captured: {num_samples}"
        message.configure(text=res)
        with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name])
    else:
        if not is_number(Id):
            res = "Enter Numeric ID"
        else:
            res = "Enter Alphanumeric Name"
        message.configure(text=res)

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Ids = [], []
    for imagePath in os.listdir("TrainingImage"):
        pilImage = Image.open(os.path.join("TrainingImage", imagePath)).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(imagePath.split('.')[1])
        faces.append(imageNp)
        Ids.append(Id)
    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    message.configure(text=res)

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    cap = cv2.VideoCapture(0)
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, im = cap.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name = df.loc[df['Id'] == Id]['Name'].values[0]  # Retrieve the first value
                tt = f"{Id}-{name}"
                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if cv2.waitKey(1) == ord('q'):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    Hour, Minute, Second = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S').split(':')
    fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName, index=False)
    cap.release()
    cv2.destroyAllWindows()
    res = "Attendance Saved"
    message2.configure(text=res)


clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg=button_bg_color, width=15, height=1,
                       activebackground="red", font=('times', 15, 'bold'))
clearButton.place(x=900, y=200)

takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="white", bg=button_bg_color, width=15, height=2,
                    activebackground="Red", font=('times', 15, 'bold'))
takeImg.place(x=350, y=550)

trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="white", bg=button_bg_color, width=15,
                     height=2, activebackground="Red", font=('times', 15, 'bold'))
trainImg.place(x=550, y=550)

trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="white", bg=button_bg_color, width=15,
                     height=2, activebackground="Red", font=('times', 15, 'bold'))
trackImg.place(x=750, y=550)

quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg=button_bg_color, width=15, height=2,
                       activebackground="Red", font=('times', 15, 'bold'))
quitWindow.place(x=950, y=550)

copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,
                    font=('times', 30, 'italic bold underline'), fg="red")
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Kaleem", "", "TEAM", "superscript")
copyWrite.configure(state="disabled", fg="red")
copyWrite.pack(side="left")
copyWrite.place(x=800, y=750)

window.mainloop()
