#!/bin/python


import cv2


def main():
    face_tracker = cv2.CascadeClassifier("frontal_face_features.xml")

    # ---------------------------------------------------------------- #
    # Use an OpenCV window to display a live feed of your webcam.
    # When you press SPACE, the program should toggle between "recording" mode and "stopped" mode.
    # * In "stopped" mode, you should only display the live feed, with no modifications.
    # * In "recording" mode, you should display a red circle on the top left corner to show that the mode is active.
    #   You should use the `face_tracker` to find any faces in the current frame, and display
    #   a green rectangle where the detected faces are (draw the outline, not a solid rectangle).

    # Your code here
    cv2.namedWindow("Webcam Feed")
    cap = cv2.VideoCapture(0)
    recording = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if recording:
            faces = face_tracker.detectMultiScale(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)

        cv2.imshow("Webcam Feed", frame)
        key = cv2.waitKey(1)
        if key == ord(" "):
            recording = not recording
        elif key == 27:
            break

    # ---------------------------------------------------------------- #

if __name__ == "__main__":
    main()
