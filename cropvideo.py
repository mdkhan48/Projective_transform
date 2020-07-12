from keras.preprocessing import image
import numpy as np
import PIL.Image
import cv2
import imutils
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array

print("[INFO] starting video stream...")
vs=cv2.VideoCapture('/home/sayem/Desktop/Agbot_video/Orig_weed_vid/1rag_1pig_separate.mp4')
#vs=cv2.VideoCapture('/home/sayem/Desktop/Agbot_video/Orig_weed_vid/2_ragweed_separate.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (240, 640)) #(width, height) has to equal to cropped frame

while True:
    # grab the frame from the threaded video stream and resize it
    ret,frame = vs.read() #for vs=cv2.VideoCapture(3)
    #height, width, channels = frame.shape
    #print('height',height)
    #print('width', width)


    #image = cv2.resize(frame, (150, 150))
    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    copy_image = image.copy()
    height, width, channels = copy_image.shape
    print('rotated height',height)
    print('rotated width', width)
    half_height = 640;
    half_width = 360;
    #image = image.astype("float") / 255.0
    #image = img_to_array(image)
    cropped_img = copy_image[0:half_height, half_width+120:720]

    fshape = cropped_img.shape;
    fheight = fshape[0];
    fwidth = fshape[1];
    print (fwidth, fheight)


    out.write(cropped_img)

    # show the output image
    #cv2.imshow("Output", copy_image)
    cv2.imshow("Cropped output", cropped_img)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

out.release()
cv2.destroyAllWindows()