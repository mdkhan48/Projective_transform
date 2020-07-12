from keras import layers
from keras import models
from keras.models import load_model
import h5py
from keras.preprocessing import image
import numpy as np
import PIL.Image
import cv2
import imutils
from keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
import time
#import serial
from imutils.video import FPS

print("[INFO] loading model...")
model = load_model('5_3_2_VGG16_gpu_1.h5')


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
#vs=cv2.VideoCapture('/home/sayem/Desktop/Agbot_video/test5_right.mp4')

vs=cv2.VideoCapture('/home/sayem/Desktop/Agbot_video/Cropped_weed_vid/Ragweed_placedRight_centerCamera.avi')

#vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


while True:
    # grab the frame from the threaded video stream and resize it
    ret,frame = vs.read() #for vs=cv2.VideoCapture(3)
    #frame = vs.read()

    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (150, 150))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    x = np.expand_dims(image, axis=0)

    preds = model.predict(x)[0]
    label = ['cocklebur','pigweed','ragweed']

    results = []
    for i in range(len(label)):
        result = [label[i] , preds[i]]
        results.append(result)
        #'results' give final detect object 'label' and 'prediction' in list format::
        # results=[['cocklebur', 0.0], ['pigweed', 0.0], ['ragweed', 1.0]]

    label = [results[0][0],results[1][0],results[2][0]]
    proba = [results[0][1],results[1][1],results[2][1]]

    label1 = "{}: {:.2f}%".format(label[0], proba[0] * 100)
    label2 = "{}: {:.2f}%".format(label[1], proba[1] * 100)
    label3 = "{}: {:.2f}%".format(label[2], proba[2] * 100)

    # update the FPS counter
    fps.update()
    fps.stop()
    totaltime = "Time running sec {:.2f}".format(fps.elapsed())
    fpscounter = "FPS: {:.2f}".format(fps.fps())

    print(totaltime, results)

# draw the label on the image

    output = imutils.resize(frame, width=320,height=640)
    cv2.putText(output, label1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.putText(output, label2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(output, label3, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(output, totaltime, (320, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(output, fpscounter, (320, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # show the output image
    cv2.imshow("Output", output)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cv2.destroyAllWindows()

