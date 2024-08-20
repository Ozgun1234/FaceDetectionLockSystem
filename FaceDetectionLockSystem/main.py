import time
from camerasystem import open_cam, photo_save
import pandas as pd

data = pd.read_csv("Database.csv")


def face_detection() -> bool:
    counter = 10

    while True:
        (result, id) = open_cam()

        if result:
            print(data.loc[data["USER_ID"] == int(id)])
            return True

        else:
            if id == "N":
                print("No face detected")
                time.sleep(1)
            else:
                counter -= 1

                if counter >= 5:
                    print("Unknown Face")

                elif counter == 1:
                    print("Your Face is Unknown Do U want to log your face? ", end=" ")
                    in1 = input("Please enter Y/n: ")

                    if in1 == "Y":
                        photo_save()

                    else:
                        print("Login system block out")
                        return False


face_detection()
