import os
import cv2
import random
import numpy as np
from PIL import Image


# def createImagesForFullData():
#     f_train = pd.read_csv('../Data/train.csv',sep=',')
#     train_X = f_train.ix[1:,1:]
#     train_Y = f_train.ix[1:,0]
#
#     for idx in range(len(list(train_Y))):
#         sample = np.array(np.uint8(train_X.iloc[idx])).reshape(28,28)
#         im = Image.fromarray(sample)
#         if os.path.exists('../Data/Images/' + str(list(train_Y)[idx])):
#             im.save('../Data/Images/' + str(list(train_Y)[idx]) + '/' + str(idx) + '.png')
#         else:
#             os.mkdir('../Data/Images/' + str(list(train_Y)[idx]))
#             im.save('../Data/Images/' + str(list(train_Y)[idx]) + '/' + str(idx) + '.png')


# Converts Pixel intensity to image
def pixelToImage(pixels,name = "temp"):
    try:
        sample = np.array(np.uint8(pixels)).reshape(28,28)
        im = Image.fromarray(sample)
        im.save('../temp/Images/' + name + '.png')
        return True
    except:
        return False

# Converts Image to Pixel Intensity Values
def imageToPixel(imgPath):
    # imgPath = extract_number(imgPath)
    pixels = np.uint8(np.asarray(Image.open(imgPath)).reshape(1,-1))
    return pixels

# Can't use this on ec2. Only for Debugging Purposes
def show_image(img):
	cv2.imshow("dst",img)
	cv2.waitKey()

# Extract number
def extract_number(imgPath, outputPath):
	#Reading the image file
	img_original = cv2.imread(imgPath)

	# Change the Original Image to Black and white, if not already
	if img_original.shape[2] == 3:
		img_bw  = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

	# Thresholding - Segmentation using simple thresholding
	ret,thresh = cv2.threshold(img_bw,220,255,0)
	#show_image(thresh)
	# Finding contours and hierarchy within them
	im,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

	digits = []

	# Finding all the relevant contours - out of all the contours based on the hierarchy
	for contour_idx in range(len(contours)):
		print "Inside Extract Number Function "+str(contour_idx)
		# Only select those contours which of level 1 in hierarchy
		if hierarchy[0][contour_idx][3] == 0:

			# Create a rectangular box for the contour
			x,y,w,h = cv2.boundingRect(contours[contour_idx])

			# Draw a rectangle on the original image
			cv2.rectangle(img_original,(x,y),(x+w,y+h),(0,255,0),2)

			# Crop the Contour
			cropped = cv2.bitwise_not(thresh[y:y+h,x:x+w])
			# show_image(cropped)
			#cropped = np.pad(cropped, [(5, ), (5, )], 'constant')
			a = cropped.shape
			height = a[0]
			width = a[1]


			_x = height if height > width else width
			_y = height if height > width else width

			square= np.zeros((_x,_y), np.uint8)

			new_squared_image_height_component_1 = (_y-height)/2
			new_squared_image_height_component_2 = _y-(_y-height)/2
			new_squared_image_width_component_1 = (_x-width)/2
			new_squared_image_width_component_2 = _x-(_x-width)/2

			width_difference = (new_squared_image_width_component_2 - new_squared_image_width_component_1) - width
			new_squared_image_width_component_1 = new_squared_image_width_component_1 + width_difference

			height_difference = (new_squared_image_height_component_2 - new_squared_image_height_component_1) - height
			new_squared_image_height_component_1 = new_squared_image_height_component_1 + height_difference

			square[new_squared_image_height_component_1:new_squared_image_height_component_2, new_squared_image_width_component_1:new_squared_image_width_component_2] = cropped

			#show_image(square)
			top_padding = 0
			for i in range(28):
				if max(square[0])>0:
					top_padding = i
					break

			square_height = square.shape[0]
			square_width = square.shape[1]

			if top_padding<square_height/5:
				new_square = np.zeros((square_width + ((square_height/5 - top_padding)*2) , square_height + ((square_height/5 - top_padding)*2)), np.uint8)
				new_square[((square_height/5 - top_padding)):square_height + ((square_height/5 - top_padding)),(square_height/5 - top_padding):square_width + (square_width/5 - top_padding)] = square

			kernel = np.ones((2,2), np.uint8)
			dilated_image = cv2.dilate(new_square,kernel)

			final_number = cv2.resize(dilated_image,(28,28),interpolation = cv2.INTER_AREA)

			kernel = np.ones((1,1), np.uint8)
			final_number = cv2.dilate(final_number,kernel)

			# final_number [final_number > 0] = 255

			#show_image(dilated_image)
			cv2.imwrite(outputPath+str(contour_idx + 1) + ".png",final_number)
			digits.append(outputPath+str(contour_idx + 1) + ".png")

	return digits
	# show_image(img_original)

# Do the Padding Analysis properly
# Create Proper Test Sets
# Create Mobile Labelled Data



# extract_number("sample.png")
# extract_number("7.jpeg")
# extract_number("8.jpeg")
# extract_number("img_3.png","./")
# extract_number("img_4.png","./")
# extract_number("img_5.png","./")
# extract_number("img_6.png","./")
# extract_number("img_7.png","./")
# extract_number("img_8.png","./")
# extract_number("img_9.png","./")
# extract_number("img_10.png","./")
#extract_number("4.jpeg")
