import cv2
import time
import numpy as np 

#save the output in a file as output.avi
# fourcc is a 4 character code
# 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

#starting the webcam and the video
#Capture function captures the video 
cap = cv2.VideoCapture(0)

#allowing the webcam to start by making the code sleep for 2 seconds 
time.sleep(2)
bg = 0

#capturing the background with 60 frames
# we need to have a video that has some seconds dedicated to the background frame so that it could easily save the background image
#we will capture background in the range of 60 0
for i in range(60):
    ret,bg = cap.read()
#flipping the background because the camera captures the image inverted  
bg = np.flip(bg,axis = 1)

#reading the captured frame until the camera is open 
while(cap.isOpened()):
    ret,img = cap.read()
    #ret returns in boolean values whether the camera is open or not 
    if not ret:
        break

    #flipping the image for consistancy 
    img = np.flip(img,axis = 1)

    #converting the color from BGR to HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #generating mask to detect red color 
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    mask_1 = cv2.inRange(hsv,lower_red,upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv,lower_red,upper_red)

    mask_1 = mask_1 + mask_2

    #open and expand the image where there is mask1 
    #we need to add effects on the colors that we have detected
    #we will be adding the diluting effect to the image in the video 
    #for that we will be using morphologyEx()
    #it accepts the following parameters(src,dst,op,kernal)
    #source,destination,an integer representing the type of morphological operation, a kernal matrix

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    #now we need to create a mask to segment out the red color from the frame
    #to do so we will be using bitwise_not()
    #Selecting only the part that does not have mask one and saving in mask 2
    mask_2 = cv2.bitwise_not(mask_1)

    #now we need to create 2 resoutions 
    #Keeping only the part of the images without the red color 
    #(or any other color you may choose)
    res_1 = cv2.bitwise_and(img, img, mask=mask_2)

    #Keeping only the part of the images with the red color
    #(or any other color you may choose)
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    #Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_output)
    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()