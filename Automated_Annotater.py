from bs4 import BeautifulSoup
from lxml import etree as ET
from object_detector import object_detector
import os
import cv2
import glob
import argparse
import time

class Auto_Annotater(object):
	
	"""docstring for Auto_Annotater"""
	def __init__(self):
		self.name = ""
		self.xmin = 0
		self.xmax = 0
		self.ymin = 0
		self.ymax = 0
		self.data = 0
		self.OD = object_detector.Object_Detector()
		
		self.input_path  = ""
		self.output_path = ""
		self.label_path  = ""
		self.pb          = ""
		self.meta        = ""
		self.threshold   = ""

		
	#@Brief: Takes arguments from the command line
	def Argument_Parser(self):
		parser = argparse.ArgumentParser(description='This program uses an object detection model ')

		parser.add_argument('--input_data', type=str,  	required=False, help = "path to image files", 		default = "test_imgs\\")
		parser.add_argument('--output_data',type=str,  	required=False, help = "path to store .xml files", 	default = "test_xml_output\\")
		parser.add_argument('--label_path', type=str, 	required=False, help = "path_of_labels.txt", 		default = "object_detector\\labels.txt")
		parser.add_argument('--pb',         type=str, 	required=False, help = ".pb file", 	default = "object_detector\\built_graph\\tiny-yolo-2c.pb")
		parser.add_argument('--meta',       type=str, 	required=False, help = "Meta file", default = "object_detector\\built_graph\\tiny-yolo-2c.meta")
		parser.add_argument('--threshold',	type=float,	required=False, help = "Meta file", default = 0.5)
		parser.add_argument('--gpu',   		type=float, required=False, help = "Meta file", default = 0.5)

		args = parser.parse_args()

		self.input_path  = args.input_data
		self.output_path = args.output_data
		self.label_path  = args.label_path
		self.pb          = args.pb
		self.meta        = args.meta
		self.threshold   = args.threshold
		self.gpu 		 = args.gpu
		self.OD.Set_TF(meta = self.meta, pb_load = self.pb, labels = self.label_path, threshold = self.threshold, gpu = self.gpu) #Inits model


#@Brief: retrieves all image names from the input folder   
	def Get_all_imgs_names_in_folder(self):
		self.List_Of_Img_Names = glob.glob(self.input_path + '*.jpg')
		self.List_Of_Img_Names.sort()

#@Brief: Loads an image from path, detects objects and structures them as objects for the given .xml file
#@Params[in]: path of image to load, xml-element to add the object annotations to.
	def Detect_Objects(self, path, data):
		self.OD.Load_Latest_Img(path)
		self.OD.Detect()

		print("xml length results = " + str(len(self.OD.results)))
		for result in self.OD.results:
			self.XML_Object(head_element = data,
							name_val = "champignon",
							pose_val = "Unspecified",
							truncated_val = 0,
							difficult_val = 0 ,
							xmin_val = result['topleft']['x'],
							ymin_val = result['topleft']['y'],
							xmax_val = result['bottomright']['x'],
							ymax_val = result['bottomright']['y'])

#@Brief: Creates top part of the annotation file containing location an image info
	def Create_Head_Of_XML(self):
		data             = ET.Element(                  'annotation')
		folder_element   = ET.SubElement(data,          'folder')
		filename_element = ET.SubElement(data,          'filename')
		path_element     = ET.SubElement(data,          'path')
		source_element   = ET.SubElement(data,          'source')
		database_element = ET.SubElement(source_element,'database')
		size_element     = ET.SubElement(data,          'size')
		width_element    = ET.SubElement(size_element,  'width')
		height_element   = ET.SubElement(size_element,  'height')
		depth_element    = ET.SubElement(size_element,  'depth')
		segmented_element= ET.SubElement(data,          'segmented')

		database_element.text = "Unknown"
		width_element.text = "640"
		height_element.text = "480"
		depth_element.text = "3"
		folder_element.text = str(os.getcwd().split('\\')[-1])              # klopt niet als er een andere folder wordt geselecteerd
		filename_element.text = "2.jpg"
		path_element.text = str(os.path.join(os.getcwd(),filename_element.text))
		segmented_element.text = "0"

		return data

#@Brief: Creates an XML-file with data annotations of objects found with tensorflow-model.
#@Params[in]:input image path, output xml path        
	def Object_Detector_Create_XML(self, img_path = "", xml_path =""):
		data = self.Create_Head_Of_XML()
		self.Detect_Objects(path = img_path, data = data)
		b_xml = ET.tostring(data, pretty_print=True)
         
		with open(xml_path, "wb") as f:
			f.write(b_xml)
		cv2.destroyAllWindows()

#@Brief: empty frame for a single object descriptor of an annotation object
#@Params[in]: tag to link to link to the rest of .xml data, name of object, truncated bool, difficult bool, difficult bool, xmin value of annotation box,
#ymin value of annotation box, xmax value of annotation box, ymax value of annotation box.
	def XML_Object(self, head_element, name_val, pose_val, truncated_val, difficult_val, xmin_val, ymin_val, xmax_val, ymax_val):
		xml_object = ET.SubElement(head_element, 'object')
		name = ET.SubElement(xml_object, 'name')
		pose = ET.SubElement(xml_object, 'pose')
		truncated = ET.SubElement(xml_object, 'truncated')
		difficult = ET.SubElement(xml_object, 'difficult')
		bnd_box = ET.SubElement(xml_object, 'bndbox')

		xmin = ET.SubElement(bnd_box, 'xmin')
		ymin = ET.SubElement(bnd_box, 'ymin')
		xmax = ET.SubElement(bnd_box, 'xmax')
		ymax = ET.SubElement(bnd_box, 'ymax')

		name.text       = str(name_val)
		pose.text       = str(pose_val)
		truncated.text  = str(truncated_val)
		difficult.text  = str(difficult_val)
        
		xmin.text = str(xmin_val)
		ymin.text = str(ymin_val)
		xmax.text = str(xmax_val)
		ymax.text = str(ymax_val)

#@Brief: combines methods of the class to iterate over all found objects by the object detector and create an .xml file of all annotations
# and saves it with the corresponding name of the image file.
#@Params[in]: path to save the Xml file to
	def Write_Xml(self, path):
		data = self.Create_Head_Of_XML()
		for result in self.Fixed_List_Of_Dicts:
			self.XML_Object(head_element = data,
							name_val = "champignon",
							pose_val = "Unspecified",
							truncated_val = 0,
							difficult_val = 0 ,
							xmin_val = result['xmin'],
							ymin_val = result['ymin'],
							xmax_val = result['xmax'],
							ymax_val = result['ymax'])
			b_xml = ET.tostring(data)
		with open(path, "wb") as f:
			f.write(b_xml)

	def Create_Xml_Name(self, img_name):
		split_file_name = img_name.split("\\")
		self.xml_name  = self.output_path + split_file_name[-1].split(".")[0] + ".xml"


	def Run_Program(self):
		self.Argument_Parser()
		# self.init_model()
		self.Get_all_imgs_names_in_folder()
		for img_name in self.List_Of_Img_Names:
			self.Create_Xml_Name(img_name)
			self.Object_Detector_Create_XML(img_path = img_name, xml_path = self.xml_name)


if __name__ == "__main__":
	AA = Auto_Annotater()
	AA.Run_Program()