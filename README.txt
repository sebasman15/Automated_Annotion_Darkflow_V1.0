README Automated_Annotater.py

DESCRIPTION:

This program helps to speed up the dreadfull slow process of image annotation, it does so by using an existing an most likely your own trained model. 
It iterates through the given input folder and creates .xml files corresponding to the image filenames.

How to load your own model:

detection models are created with a fork of tensorflow called darkflow avalaible at: https://github.com/thtrieu/darkflow

after following the instructions on the github page you should have darkflow installed globally + the dependencies. If the demo on the github page can be run means it has been done succesfully. follow the rest of the instructions to create a model
and convert the model to a protobuff (.pb) and META-file(.meta). Place these files in the folder 'built_graph/' 

Usage included model:

$ python Automated_Annotater --input_data "test_img" --output_data "test_xml_data" --label_path "object_detector" --pb "object_detector/built_graph/tiny-yolo-2c.pb" --meta "object_detector\\built_graph\\tiny-yolo-2c.meta" --threshold 0.5 --gpu 0.5

Expected result:

When running the commandline above all the images in the folder 'test_img' will be annotated and the annotation files will be stored in the 'test_xml'

