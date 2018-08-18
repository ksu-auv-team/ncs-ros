from __future__ import division

from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import rospy
import cv_bridge
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import ros_numpy
import os

dim=(300,300)
IMAGES_DIR = 'images/'

# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************

LABELS = ("background", "path_marker", "start_gate", 
            "channel", "claw", "die1", 'die2', 'die5', 'die6',
            'roulette_wheel', 'red_wheel_side', 'black_wheel_side',
            'slot_machine', 'slot_handle', 'r_slot_target', 'y_slot_target',
            'r_ball_tray', 'g_ball_tray', 'floating_area', 'r_funnel',
            'y_funnel', 'g_chip_dispenser', 'g_chip_plate', 'dieX', 'g_funnel')

# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    img = cv2.resize(src, dim)

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    # img = img * 0.007843
    img = img/256
    return img

# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, ssd_mobilenet_graph, inputfifo, outputfifo):

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    resized_image = preprocess_image(image_to_classify)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    ssd_mobilenet_graph.queue_inference_with_fifo_elem(inputfifo, outputfifo, resized_image.astype(numpy.float32), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = outputfifo.read_elem()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))
    boxes = [num_valid_boxes]

    for box_index in range(num_valid_boxes):
            base_index = 7 + box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue

            # clip the boxes to the image size incase network returns boxes outside of the image
            x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[0]))
            y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[1]))
            x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[0]))
            y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[1]))

            x1_frac = x1 / image_to_classify.shape[0]
            y1_frac = y1 / image_to_classify.shape[1]
            x2_frac = x2 / image_to_classify.shape[0]
            y2_frac = y2 / image_to_classify.shape[1]

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            boxes.extend([output[base_index + 1], output[base_index + 2]*100, x1_frac, y1_frac, x2_frac, y2_frac])

            print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                  'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                  'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')

            # overlay boxes and labels on the original image to classify
            overlay_on_image(image_to_classify, output[base_index:base_index + 7])

            return boxes


# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info):

    # the minimal score for a box to be shown
    min_score_percent = 60

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

def img_callback():
    pass

# This function is called from the entry point to do
# all the work of the program
def main():
    # name of the opencv window
    cv_window_name = "SSD MobileNet - hit any key to exit"

    #start publisher

    rospy.init_node('SSD_node', anonymous=True)
    
    pub = rospy.Publisher('ssd_output', Float32MultiArray, queue_size=2)

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.open()

    graph = mvnc.Graph('graph1')

    # The graph file that was created with the ncsdk compiler
    graph_file_name = '/home/owl/src/ncs-ros/graph'

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    inputfifo, outputfifo = graph.allocate_with_fifos(device, graph_in_memory)

    cam= cv2.VideoCapture('/home/owl/src/videos/videos/GOPR0286.MP4')
    if not cam.isOpened():
        print('error: camera not opened')
    #below property not supported by this opencv implementation.
    # cam.set(cv2.CAP_PROP_BUFFERSIZE , 1)
    
    #images for testing
    # for filename in os.listdir('./images'):
    #     image = cv2.imread('images/' + filename)
    #     print(filename)
    #     # image = cv2.imread('images/img00000.jpg')
    #     output = run_inference(image, graph, inputfifo, outputfifo)
    #     #message will be in format described above
    #     msg = Float32MultiArray()
    #     msg.data = output
    #     print(output)
    #     pub.publish(msg)

    #     cv2.imshow(cv_window_name, image)
    #     cv2.waitKey(0)

    img_counter = 0
    while not rospy.is_shutdown():
        ret, image = cam.read()
        print(img_counter)
        img_counter += 1
        #run network
        if img_counter % 3 == 0:
            output = run_inference(image, graph, inputfifo, outputfifo)
            #message will be in format described above
            msg = Float32MultiArray()
            msg.data = output
            print(output)
            pub.publish(msg)

            cv2.imwrite('out_imgs/'+str(img_counter)+'.jpg', image)

            # display the results
            cv2.imshow(cv_window_name, image)
            cv2.waitKey(1)




    # Clean up the graph and the device
    graph.destroy()
    inputfifo.destroy()
    outputfifo.destroy()
    device.close()
    device.destroy()

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    main()
