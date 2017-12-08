"""
Download the vqa data and preprocessing.
Version: 1.0
Contributor: Jiasen Lu
"""

# Download the VQA Questions from http://www.visualqa.org/download.html
import json
import os 
import argparse

def download_vqa():
 
  # Downlaod the VQA Questions :
  os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip -P zip/')
  os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip -P zip/')
  os.system('wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip -P zip/')
  
  # Download the VQA Annotations :
  os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Train_mscoco.zip -P zip/')
  os.system('wget http://visualqa.org/data/mscoco/vqa/Annotations_Val_mscoco.zip -P zip/')
  
  # Let us now unzip the annotations :
  os.system('unzip zip/Questions_Train_mscoco.zip -d annotations/')
  os.system('unzip zip/Questions_Val_mscoco.zip -d annotations/')
  os.system('unzip zip/Questions_Test_mscoco.zip -d annotations/')
  os.system('unzip zip/Annotations_Train_mscoco.zip -d annotations/')
  os.system('unzip zip/Annotations_Val_mscoco.zip -d annotations/')
  
  def main(params):
    if params['downlaod'] == 1:
      download_vqa 
      
      # Note the following :
    '''
    Put the VQA data into single  JavaScript Object Notation or JSON file, where [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ]
    '''
    
    


  
