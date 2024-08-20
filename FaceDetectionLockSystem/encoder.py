import cv2
import face_recognition
import pickle
import os


# Import all the Images

def save_encodings() -> None:
    start_encoding()
    file = open("EncodeFile.p", "wb")
    pickle.dump(start_encoding(), file)
    file.close()


def find_encoding(p_img_list: list) -> list:
    encode_list = []

    for img in p_img_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]  # Find the face encodings
        encode_list.append(encode)
    return encode_list


def start_encoding() -> list:
    folder_path = "Images"
    mode_path_list = os.listdir(folder_path)

    # Getting User Id and User Images
    img_list = []
    user_ids = []

    for path in mode_path_list:
        # Getting user image
        img_list.append(cv2.imread(os.path.join(folder_path, path)))

        # Getting User Id
        user_ids.append(path.split(".")[0])

    encode_list_known = find_encoding(img_list)
    encoding_list_known_with_ids = [encode_list_known, user_ids]

    return encoding_list_known_with_ids


save_encodings()
