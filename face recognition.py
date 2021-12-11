import face_recognition
import cv2
import numpy as np
import csv
import pickle
import pyttsx3

def att(name):
    with open('Attendance.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile)
        #now = datetime.now()
        #dtString = now.strftime('%H:%M')
        #csvFile.writelines(f'{[name]},{dtString}')
        writer.writerow([name])
        csvFile.close()


f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)
f.close()

f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)
f.close()
known_face_encodings = []
known_face_names = []



for ref_id , embed_list in embed_dictt.items():
    for my_embed in embed_list:
        known_face_encodings +=[my_embed]
        known_face_names += [ref_id]
video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for encodeFace, faceLoc in zip(face_encodings,face_locations):

            matches = face_recognition.compare_faces(known_face_encodings, encodeFace)

            face_distances = face_recognition.face_distance(known_face_encodings, encodeFace)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index].upper()

            else:
                name = "unknown get enrolled"
            face_names.append(name)
            top_s, right, bottom, left = faceLoc
            top_s *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top_s), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            att(ref_dictt[name])
                #p = pyttsx3.init()
                #p.say("you are" + ref_dictt[name])
                #p.runAndWait()



    cv2.imshow('Video', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
