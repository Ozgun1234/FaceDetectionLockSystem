import cv2
import face_recognition
import pickle
import numpy as np
from encoder import save_encodings
import sys
import os

#
cap = cv2.VideoCapture(1)

cap.set(3, 640)
cap.set(4, 480)


def open_cam():
    save_encodings()
    # print("Loading Encoding File is Started")
    file = open("EncodeFile.p", "rb")
    encoding_list_known_with_ids = pickle.load(file)
    file.close()
    # print("Loading Encoded File End")

    encode_list_known, user_ids = encoding_list_known_with_ids
    # print(user_ids)

    while True:
        _, img = cap.read()

        (result, id) = face_recognition_f(img, encode_list_known, user_ids)

        # cv2.imshow("Face Detection", img)

        # key = cv2.waitKey(1)

        # if key == ord("q"):
        # break

        if result:
            return result, id
        else:
            if id == "N":
                return False, "N"
            return False, None


def face_recognition_f(a_img, a_encode_list_known, a_user_ids):
    img_s = cv2.resize(a_img, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    # Detect Faces
    face_cur_frame = face_recognition.face_locations(img_s)
    encode_cur_frame = face_recognition.face_encodings(img_s, face_cur_frame)

    if face_cur_frame:

        for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):

            matches = face_recognition.compare_faces(a_encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(a_encode_list_known, encode_face)

            # print("matches:", matches)
            # print("face_dis:", face_dis)

            matchIndex = np.argmin(face_dis)
            # print("Match Index: ", matchIndex)

            if matches[matchIndex]:
                id = a_user_ids[matchIndex]
                return True, id
            else:
                return False, None

    else:
        return False, "N"


def restart_program():
    """Restarts the current program.
  Note: this function does not return. Any cleanup action (like
  saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, *sys.argv)


def photo_save():
    a = input("Please create a user id: ")
    image_path = "Images"

    while True:
        success, img = cap.read()

        if success:

            cv2.imwrite(os.path.join(image_path, "%s.jpg") % a, img)
            print("Photo Saved")
            cv2.destroyAllWindows()
            # restart_program()
            break
        else:
            print("Camera cannot turn on")
            break
