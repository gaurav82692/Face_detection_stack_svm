'''
Befor running this script please do download celebrity2000.mat file and images dataset from
https://bcsiriuschen.github.io/CARC/
Paste both images data set and celebrity2000.mat files in same folder in which you are running script.
folder name of images dataset should be 'CACD2000\\CACD2000' and it should contain all downloaded images.
'''
FILE_PATH='celebrity2000.mat'
import h5py
import numpy as np
import pandas as pd
import shutil
import time
from matplotlib import pyplot as plt
import datetime
from PIL import Image
from matplotlib import cm
import os
from info_logging import *
def processing_mat(FILE_PATH):
 log("Script started")
##Processing Started for celebrityData Structure##
 log("Started Processing for celebrityData structure from DataSet")
 arrays_celebrityData = {}
 f = h5py.File(FILE_PATH,'r')
 for k, v in f['celebrityData'].items():
        if k in ['identity','birth']:
            arrays_celebrityData[k]=np.array(v).ravel()
 names=np.array([])
 for i in range(len(f['celebrityData/name'][0])):
  fetch_name = f['celebrityData/name'][0][i]
  obj = f[fetch_name]
  name_val = ''.join(chr(i) for i in np.array(obj[:]).ravel())
  names=np.append(names,name_val)
 arrays_celebrityData['names'] = names.ravel()
 DF_celebrityData=pd.DataFrame(arrays_celebrityData)
 #DF_celebrityData.to_csv("celebrityData.csv") ##uncomment if you want to save this csv to disk
 log("Succesfully Processed celebrityData structure from DataSet")
 log("Started Processing for celebrityImageData structure from DataSet")
################Processing Ended for celebrityData Structure######################
################Processing Started for celebrityImageData Structure######################
 arrays_celebrityImageData = {}
 f2 = h5py.File(FILE_PATH,'r')
 for k, v in f2['celebrityImageData'].items():
        if k in ['identity','age','rank']:
            arrays_celebrityImageData[k]=np.array(v).ravel()

 file_paths_src=np.array([])
 for i in range(len(f['celebrityImageData/name'][0])):
  fetch_file_name = f['celebrityImageData/name'][0][i]
  obj = f[fetch_file_name]
  file_path_val = ''.join(chr(i) for i in np.array(obj[:]).ravel())

  file_paths_src=np.append(file_paths_src,'CACD2000\\CACD2000\\'+file_path_val)
 arrays_celebrityImageData['src_file_path'] = file_paths_src.ravel()
 DF_celebrityImageData=pd.DataFrame(arrays_celebrityImageData)
 #DF_celebrityImageData.to_csv("celebrityImageData.csv") ##uncomment if you want save this csv data to disk
 log("Succesfully Processed celebrityImageData structure from DataSet")
################Processing Ended for celebrityImageData Structure######################
################Processing Started for joining celebrityImageData & celebrityData Structure based on identity column as key column in join####
 log("Joining Dataframes")
 df_inner = pd.merge(DF_celebrityData, DF_celebrityImageData, on='identity', how='inner')
 df_inner=df_inner.iloc[:,2:]
 # deleting both data frames to prevent memory overflow as we #have already merged the datasets in df_inner
 del DF_celebrityData
 del DF_celebrityImageData
 log("Starting Restructuring Folder Images")
 #Classifying images in CACD2000\\CACD2000 in respective celebrity name folders
 file_paths_dest = np.array([])
 for name in df_inner['names'].unique():
  os.mkdir('CACD2000\\CACD2000\\'+str(name))
  x=df_inner[df_inner['names']==name]
  for file in x['src_file_path']:
   shutil.move(file, 'CACD2000\\CACD2000\\'+name)
   file_paths_dest = np.append(file_paths_dest, 'CACD2000\\CACD2000\\' + str(name)+"\\"+str(file.split('\\')[-1]))
 df_inner['file_paths_dest']=file_paths_dest

 log("Completed Restructuring Folder Images")
 df_inner.to_csv('DataSet_Final.csv')
 log("Script Ended")

if __name__=='__main__':
 processing_mat(FILE_PATH)



