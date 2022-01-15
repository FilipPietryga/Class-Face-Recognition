import cv2
import numpy as np
import face_recognition
import os

# source of students images
path = "students"

# self explanatory
studentsImages = []

#s elf explanatory
studentsNames = []
studentsSurnames = []

# check every file in the directory and load image from it
# then scrap each students name and surname
for image in os.listdir(path):
    currentImage = cv2.imread(path + "/" + image)
    studentsImages.append(currentImage)
    fullname = os.path.splitext(image)[0]
    studentsNames.append(fullname[0:fullname.index("_")])
    studentsSurnames.append(fullname[fullname.index("_")+1:])

# scrap encoding for each student's image
def scrapEncodingData(studentsImages):
    imagesEncodings = []
    
    #for each image
    for image in studentsImages:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imagesEncodings.append(face_recognition.face_encodings(img)[0])
        
    return imagesEncodings

# self explaining
def markAttendance(studentName, studentSurname):
    file = open("attendance.csv", "r+")
    header = file.readline()
    print(header)
    marked = []
    while True:
        line = file.readline()
        if not line:
            break
        print(line)
        entry = line.split(", ")
        if(entry[1] == '\n'):
            entry[1] = entry[1][:-1]
        marked.append((entry[0], entry[1]))
    print(marked)
    if (studentName, studentSurname) not in marked:
        file.write("\n" + studentName + ", " + studentSurname);
        
# obtain the encoding for every student
studentsEncodings = scrapEncodingData(studentsImages)

# analyze the class picture looking for
# students faces and mark attendance of
# these that are located
def analyzePicture(image, encodings):
    #resize the image
    img = cv2.resize(image, (0,0), None, 0.25, 0.25)
    
    #convert the color standard
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(image, faceCurFrame)
    
    for itemEncoding in encodeCurFrame:
        
        distance = face_recognition.face_distance(encodings, itemEncoding)
        leastDistance = np.argmin(distance)
        name = studentsNames[leastDistance].capitalize()
        surname = studentsSurnames[leastDistance].capitalize()
        markAttendance(name, surname)
    
attendees = face_recognition.load_image_file("class.jpg")
    
analyzePicture(attendees, studentsEncodings);
    