import os  # For os operations
from pdf2image import convert_from_path  # Convert pdf to image
import sys  # For taking argument
import shutil


## This python program takes input parameter PDF FILE and converts that into corresponding images ####
# The images are stored in "Images" Folder
# for running the script run python3 FileName.py sample.pdf
# Replace smaple.pf with path of pdf file


try:
    shutil.rmtree('Images')
    file_name = sys.argv[1]
    os.mkdir('Images')
    images = convert_from_path(file_name, output_folder='Images', fmt='jpg')
except FileNotFoundError:
    file_name = sys.argv[1]
    os.mkdir('Images')
    images = convert_from_path(file_name, output_folder='Images', fmt='jpg')

image_names = os.listdir('Images')
image_names.sort()
for i in range(len(image_names)):
    os.rename('Images/'+image_names[i], 'Images/'+str(i+1)+'.jpg')
image_names = os.listdir('Images')
print(image_names)
