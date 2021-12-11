import cv2
import face_recognition
import speech_recognition as sr
import pickle
import pyttsx3
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    print("look at the cam ,please!")
    engine.say("look at the cam ,please!")
    engine.runAndWait()
def forerror(command):
    t = pyttsx3.init()
    voices = t.getProperty('voices')
   # t.setProperty('voice', voices[1].id)
    t.say("say again please")
    t.runAndWait()
s_no=input("serial number: ")
r = sr.Recognizer()
#ref_id=input("serial number")
#df=pd.read_csv("Attendance.csv")
try:
    with sr.Microphone() as source2:

        # wait for a second to let the recognizer
        # adjust the energy threshold based on
        # the surrounding noise level
        r.adjust_for_ambient_noise(source2, duration=1)
        # listens for the user's input
        audio2 = r.listen(source2)
        # Using google to recognize audio
        MyText = r.recognize_google(audio2)
        MyText = MyText.lower()

        # writing to csv file

        print("OK ," + MyText)
        SpeakText(MyText)
        #presentlist.append([MyText])
        #filename = "Attendance.csv"

except sr.RequestError as e:
    print("Could not request results; {0}".format(e))

except sr.UnknownValueError:
    forerror("command")
    print("say again please")
    exit()
ref_id=MyText
try:
    f=open("ref_name.pkl","rb")

    ref_dictt=pickle.load(f)
    f.close()
except:
    ref_dictt={}


ref_dictt[s_no]=ref_id

f=open("ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()

try:
    f=open("ref_embed.pkl","rb")

    embed_dictt=pickle.load(f)
    f.close()
except:
    embed_dictt={}

for i in range(1):
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:

        check, frame = webcam.read()

        cv2.imshow("Capturing", frame)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        key = cv2.waitKey(1)

        if key == ord('s'):
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations != []:
                face_encoding = face_recognition.face_encodings(frame)[0]
                if ref_id in embed_dictt:
                    embed_dictt[s_no] += [face_encoding]
                else:
                    embed_dictt[s_no] = [face_encoding]
                webcam.release()
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
f=open("ref_embed.pkl","wb")
pickle.dump(embed_dictt,f)
f.close()