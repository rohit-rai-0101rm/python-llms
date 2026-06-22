print("Program started")

import speech_recognition as sr


def main():
    print("Inside main")

    r=sr.Recognizer()
    print("Before microphone")


    with sr.Microphone() as source:
        print("Microphone opened")

        r.adjust_for_ambient_noise(source)
        r.pause_threshold=2
        print("Speak something...")
        audio=r.listen(source)

        print ("Processing audio ...STT")
        stt=r.recognize_google(audio)

        print ("You said",stt)



main()
