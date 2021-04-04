import cv2
import numpy as np
import os
import sys
import json
import shutil
import time


# For running this file command : python3 ImageDictonary.py

# Detetct tables and cells within table. The output is save in json format in file my.json
# Need to have images in "Images" folder'

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# Detects table coordinates


def box_extraction(img_for_box_extraction_path):

    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    cv2.imwrite('Sample.jpg', img)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin.jpg", img_bin)

##########################################################################################################
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//100
    # Define proper kernel length

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    #cv2.imwrite("verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    #cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes containing only one
    #cv2.imwrite("img_final_bin", img_final_bin)
    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    shape = img.shape
    page_width = shape[1]
    page_height = shape[0]
    print(shape)
    dictionary_mapped = {}
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater then 50 and width greater than 300
        # Need to change the condition with respect to table dimensions in page
        if (w > 300 and h > 50 and (0.8*page_width > w or 0.8*page_height > h)):
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite('sample.png', new_img)
            returned_smaller_boxes = samller_box_extraction('sample.png')
            if (len(returned_smaller_boxes) > 1):
                dictionary_mapped[str(y)+"," + str(y+h)+"," +
                                  str(x)+"," + str(x+w)] = returned_smaller_boxes

    return (dictionary_mapped)

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./Temp/img_contour.jpg", img)

# Detect cells in table


def samller_box_extraction(img_for_box_extraction_path):

    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    #cv2.imwrite("Image_bin.jpg", img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//80

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    #cv2.imwrite("verticle_lines.jpg", verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    #cv2.imwrite("horizontal_lines.jpg", horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(
        verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(
        img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    #cv2.imwrite("img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    answer = []
    shape = img.shape
    page_width = shape[1]
    page_height = shape[0]

    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)

        # If the box height is greater than 16 and width is greater than 16
        # Need to change the height and width according to cells in table or otherwise numbers will also be enclosed in box
        if (w > 16 and h > 16 and (0.8*page_width > w or 0.8*page_height > h)):
            idx += 1
            answer.append([y, y+h, x, x+w])
    return (answer)

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./Temp/img_contour.jpg", img)


image_directory = 'Images'
image_names = os.listdir('Images')
print(image_names)
main_dictionary = {}
for i in image_names:
    r = box_extraction("/home/piyushonkar/WorkingTableExtraction/Images/" + i)
    main_dictionary[i] = r
ticks = time.time()
for i in image_names:
    img = cv2.imread("/home/piyushonkar/WorkingTableExtraction/Images/" + i)
    value_of_page = main_dictionary[i]
    for key, value in value_of_page.items():
        r = list(map(int, key.split(",")))
        y1 = r[0]
        y2 = r[1]
        x1 = r[2]
        x2 = r[3]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for cells in value:
            cell_y1 = cells[0]
            cell_y2 = cells[1]
            cell_x1 = cells[2]
            cell_x2 = cells[3]
            img = cv2.rectangle(img, (x1+cell_x1, y1+cell_y1),
                                (x1+cell_x2, y1+cell_y2), (255, 0, 0), 1)
    cv2.imwrite("NewImage"+str(ticks)+i, img)

#################################################################################

# image_dictionary format = {"pageno":"upper_left_y,lower_right_y,upper_left_x,lower_right_x":[y1,y2,x1,x2]}
# y1 y2 x1 and x2 are coordinates relative to the table . To get exact coordinates add upper_left and upper_right


# Save to json Format
json_format = {}
for key1, value1 in main_dictionary.items():
    tables = 1
    table_dictionary = {}
    for value3, value4 in value1.items():
        r = list(map(int, value3.split(",")))
        y1 = r[0]
        y2 = r[1]
        x1 = r[2]
        x2 = r[3]
        cells = 1
        cells_dictionary = {}
        for cell_value in value4:
            uly = cell_value[0]
            lry = cell_value[1]
            ulx = cell_value[2]
            lrx = cell_value[3]
            small_cell_dictionary = {}
            small_cell_dictionary["uly"] = uly
            small_cell_dictionary["lry"] = lry
            small_cell_dictionary["ulx"] = ulx
            small_cell_dictionary["lrx"] = lrx
            cells_dictionary["cell:"+str(cells)] = small_cell_dictionary
            cells = cells + 1
        intermediate_dictionary = {}
        intermediate_dictionary["uly"] = y1
        intermediate_dictionary["lry"] = y2
        intermediate_dictionary["ulx"] = x1
        intermediate_dictionary["lrx"] = x2
        intermediate_dictionary["cells"] = cells_dictionary
        table_dictionary["Table:"+str(tables)] = intermediate_dictionary
        tables = tables + 1
    json_format[key1] = table_dictionary


with open("my.json", "w") as f:
    json.dump(json_format, f, indent=4)


#####################################################################
# Removing Intermediate Files and Folders
os.remove('sample.png')
shutil.rmtree('Images')
########################################################
