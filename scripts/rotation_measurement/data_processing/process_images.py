import cv2, glob
from blob_detector import AngleDetector

l = glob.glob("./wrote_ims/img_temp_*")
l.sort(key=lambda x : int(x.split("_")[3]))

AD = AngleDetector(cv2Image=True, writeImages=False)
with open("output.csv", "w") as angle_file:
    for image in l:
        i = cv2.imread(image)
        AD.update_angle(i)
        
        print(AD.getAngle())
        cv2.imshow("image", i)
        if AD.getAngle() > 52:
            cv2.waitKey(0)
        else:
            cv2.waitKey(0)
        print(image)
        angle_file.write(str(AD.getAngle()) + "\n")