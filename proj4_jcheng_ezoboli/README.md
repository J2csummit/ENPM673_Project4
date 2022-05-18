Project 4
Author: Justin Cheng and Enrico Zoboli
Class: ENPM 673

Task
----
This project uses YOLO and MaskRCNN libraries to identify objects.

Used Libraries
--------------
scipy, numpy, matplotlib, pandas, statsmodels, sklearn, scikit-image, theano, tensorflow, keras

Steps to run code
-----------------
1. Ensure that you have Python 3 installed
	- a) Code was written using Python 3.7
2. Extract zip file into any directory
3. Make sure that all libraries are installed.
	- a) For instruction on how to install libraries, refer to https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
4. Download the following into the project directory: https://pjreddie.com/media/files/yolov3.weights
5. Navigate to the directory of the project and run model.py python
	- a) Run from terminal
6. Navigate to the directory of the project and run yolo.py python
	- a) Run from terminal
5. yolo.py python file should output png files in the directory that show identified objects in original images
6. In project directory, from terminal, run "git clone https://github.com/alsombra/Mask_RCNN-TF2.git"
	- a) After cloning, there should be a folder titled "Mask_RCNN-TF2"
7. Download the following file into "Stuff to go into Mask_RCNN-TF2" folder: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
	- a) rename this file "model.h5"
8. Copy contents of "Stuff to go in Mask_RCNN-TF2" folder into "Mask_RCNN-TF2" folder that was just created.
	- a) do not simply drag one folder into the other, you must actuallly place the contents of one folder in the other folder
9. Navigate to the directory of "Mask_RCNN-TF2" in the project folder and run yolo.py python "m_rcnn.py"
10. m_rcnn.py python file should output png files in the directory that show identified objects in original images

