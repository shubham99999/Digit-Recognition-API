#!pip install opencv-python
#!apt update && apt install -y libsm6 libxext6
import os
import cv2
import sys
sys.path.append('../')
import random
import pandas as pd
import numpy as np
from PIL import Image


# We are using the MNIST Data from the train.csv file and we can use this following function
# to convert the full data in image format.
def createImagesForFullData():
    f_train = pd.read_csv('../Data/train.csv',sep=',')
    train_X = f_train.ix[1:,1:]
    train_Y = f_train.ix[1:,0]

    for idx in range(len(list(train_Y))):
        sample = np.array(np.uint8(train_X.iloc[idx])).reshape(28,28)
        im = Image.fromarray(sample)
        if os.path.exists('../Data/Images/' + str(list(train_Y)[idx])):
            im.save('../Data/Images/' + str(list(train_Y)[idx]) + '/' + str(idx) + '.png')
        else:
            os.mkdir('../Data/Images/' + str(list(train_Y)[idx]))
            im.save('../Data/Images/' + str(list(train_Y)[idx]) + '/' + str(idx) + '.png')

# Using this function we can convert pixel information of any mnist image to actual png image
def pixelToImage(pixels,name = "temp"):
    try:
        sample = np.array(np.uint8(pixels)).reshape(28,28)
        im = Image.fromarray(sample)
        im.save('../temp/Images/' + name + '.png')
        return True
    except:
        return False


# Using this function we can convert any image to it's corresponding pixel intensity array.
def imageToPixel(imgPath):
    # imgPath = extract_number(imgPath)
    pixels = np.uint8(np.asarray(Image.open(imgPath)).reshape(1,-1))
    return pixels

# Can't use this on ec2. Only for Debugging Purposes
def show_image(img):
	cv2.imshow("dst",img)
	cv2.waitKey()

# This function extracts different parts of the image (in our case different Digits)
def extract_number(imgPath, outputPath):
	#Reading the image file
	img_original = cv2.imread(imgPath)

	# print "Original Image"
	# show_image(img_original)

	# Change the Original Image to Black and white, if not already
	if img_original.shape[2] == 3:
		img_bw  = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

	# print "Black and White Image"
	# show_image(img_bw)

	# Thresholding - Segmentation using simple thresholding
	ret,thresh = cv2.threshold(img_bw,220,255,0)

	# print "Thresholding Result"
	# show_image(thresh)



	# Below given function call is Finding contours and hierarchy within them.
	# Reason why we are using RETR_TREE - Because that makes us easy for us to just extract the numbers
	# from the images because those will mostly always be at the same hierarchy. It also helps us in excluding other
	# contours from different heirarchy levels which are captured.
	# Reason why we are using CHAIN_APPROX_SIMPLE approximation method is its enough for us to get the digit out of the image
	# and because it only return 4 corners of the boundary it's easy on the memory as well. In more complex situations
	# where the part of the image which is to be extracted is more complicated in terms of area capture CHAIN_APPROX_NONE.
	im,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

	digits = []
	images = []

	# Finding all the relevant contours - out of all the contours based on the hierarchy
	for contour_idx in range(len(contours)):
		# Only select those contours which of level 1 in hierarchy
		if hierarchy[0][contour_idx][3] == 0:

			# Create a rectangular box for the contour
			x,y,w,h = cv2.boundingRect(contours[contour_idx])

			# Draw a rectangle on the black and white image - to mark where is the
			# digit in the image
			cv2.rectangle(img_bw,(x,y),(x+w,y+h),(0,255,0),2)

			# print "Contour Detection Result"
			# show_image(img_bw)

			# Crop the Contour - to just get the digit area from the image
			cropped = cv2.bitwise_not(thresh[y:y+h,x:x+w])

			# print "Cropped Digit"
			# show_image(cropped)

			# Get the shape (dimensions of the cropped image i.e the digit)
			a = cropped.shape
			height = a[0]
			width = a[1]

			####################################################################
			# Below given code performs the squaring of the digit image.
            # The image of the digit we get till the last section of the code is not exactly square
            # Where as each image that we have in mnist dataset on which we trained our model is square (28by28 pixel).
            # So to make our image more compatible with our model we make it exact square by centering the digit and adding
            # extra black pixels on one dimension (may it be height or width, which ever is less)

			_x = height if height > width else width
			_y = height if height > width else width

			square= np.zeros((_x,_y), np.uint8)


			new_squared_image_height_component_1 = (_y-height)//2
			new_squared_image_height_component_2 = _y-(_y-height)//2
			new_squared_image_width_component_1 = (_x-width)//2
			new_squared_image_width_component_2 = _x-(_x-width)//2

			width_difference = (new_squared_image_width_component_2 - new_squared_image_width_component_1) - width
			new_squared_image_width_component_1 = new_squared_image_width_component_1 + width_difference

			height_difference = (new_squared_image_height_component_2 - new_squared_image_height_component_1) - height
			new_squared_image_height_component_1 = new_squared_image_height_component_1 + height_difference

			square[new_squared_image_height_component_1:new_squared_image_height_component_2, new_squared_image_width_component_1:new_squared_image_width_component_2] = cropped

			####################################################################

			# print "Squared Image"
			# show_image(square)

            ####################################################################
            # Below given code is used to add padding to the top and bottom of the digit image
            # so that the digit itself is not touching the boundaries of the image. This is
            # done with maintaining the aspect ratio of digit so the information loss and distortion is
            # minimal. Before this code executes the digit was always sticking to the top and bottom boundary
            # but once this part executes the gap is increased.
			top_padding = 0
			for i in range(28):
				if max(square[0])>0:
					top_padding = i
					break

			square_height = square.shape[0]
			square_width = square.shape[1]

			if top_padding<square_height/5:
				new_square = np.zeros((square_width + ((square_height//5 - top_padding)*2) , square_height + ((square_height//5 - top_padding)*2)), np.uint8)
				new_square[((square_height//5 - top_padding)):square_height + ((square_height//5 - top_padding)),(square_height//5 - top_padding):square_width + (square_width//5 - top_padding)] = square

            ####################################################################

            # print "New Squared Image with padding"
			# show_image(new_square)

            # Image is resized and Dilated
			kernel = np.ones((2,2), np.uint8)
			dilated_image = cv2.dilate(new_square,kernel)

			final_number = cv2.resize(dilated_image,(28,28),interpolation = cv2.INTER_AREA)

			# print "Resized Image"
			# show_image(final_number)

            # Once again is Image Dilated after the resizing to increase the stroke width
			kernel = np.ones((1,1), np.uint8)
			final_number = cv2.dilate(final_number,kernel)

            # Final image is segmented to contain only black and white pixel.
			final_number [final_number > 0] = 255

			# print "Black and White Dilated Resized Image"
			# show_image(final_number)

			# show_image(final_number)

            # Finally we write this digit image into the persistent memmory (Hard disk) and append the path to this image
            # in digits list.
			cv2.imwrite(outputPath+str(contour_idx + 1) + ".png",final_number)
			digits.append(outputPath+str(contour_idx + 1) + ".png")

	return digits

if __name__ == '__main__':
	extract_number("sampleImageFromAndroidApp.png","./Intermediate/")
