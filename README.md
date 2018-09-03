# Instructions
* Clone this Github toolkit for caffe: https://github.com/chuanqi305/MobileNet-SSD
* "Caffe implementation of Google MobileNet SSD detection network, with pretrained weights on VOC0712 and mAP=0.727."
## Get the dataset
* Download the dataset you're using from the labeling tool and construct it with ` $package_dataset.py\`
* Labeling Tool can be found here: https://ksu-auv-team.github.io/labeling.html
* Convert the dataset from .npz to lmdb with ` $convert_dataset.py`, `$ make_data_list.py`, and `$ create_data.sh` (in that order)
* Write a label_map.prototxt that matches the dataset

## Train the network
* Use label_map.prototxt and the lmdb to train the network with the MobileNet-SSD files. First use `$ genmodel.sh <number of classes>` (usually number of folders to represent number of objects) to generate the model, then `$ train.sh` to train
* The output .caffemodel file will be in the MobileNet-SSD/snapshot folder

## Deploy the network
* Use `$ merge_bn.py` to create a deploy .prototxt and .caffemodel. It uses the .prototxt from the example folder by default. The .caffemodel file name needs to be set manually.
* Copy the deploy .prototxt and .caffemodel to the Movidius folder, then run them with `$ run-nnet.py`

## Other files
* camera_node.py publishes webcam output as a series of images
* i-listener.py subscribes to an old version of the network's output

Requires ROS, OpenCV, Caffe, and the MVNC API
