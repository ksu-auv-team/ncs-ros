import os
from shutil import copyfile

for filename in os.listdir('qual_out_imgs'):
    output_string = ('prepped_vid_images/')
    for i in range (6 - (len(filename) - 4)):
        output_string += '0'
    output_string += filename
    copyfile('qual_out_imgs/' + filename, output_string)
    print('done with ' + filename)