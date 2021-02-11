import os
PATH="Image_DataSet_100_UnProcessed"
DIR_NAMES = os.listdir(PATH)
from matplotlib import pyplot as plt
count=1
X_Axis=[]
Y_Axis=[]
for sub_folder_name in DIR_NAMES:
 FILE_NAMES = os.listdir(PATH+"\\"+sub_folder_name)
 X_Axis.append(sub_folder_name)
 Y_Axis.append(len(FILE_NAMES))
plt.plot(X_Axis, Y_Axis,label="Total Images")
plt.title("Images Per Class 90:10 Ratio")
plt.xticks(rotation=90,fontsize=5.5)
plt.ylabel("No. of Images")


TEST_PATH="processed_dataset\\test"
TEST_DIR_NAMES = os.listdir(TEST_PATH)
count=1
X_Axis_TS=[]
Y_Axis_TS=[]
for sub_folder_name in TEST_DIR_NAMES:
 TEST_FILE_NAMES = os.listdir(TEST_PATH+"\\"+sub_folder_name)
 
 X_Axis_TS.append(sub_folder_name)
 Y_Axis_TS.append(len(TEST_FILE_NAMES))
plt.plot(X_Axis_TS, Y_Axis_TS,label="Test Images")



TRAIN_PATH="processed_dataset\\train"
TRAIN_DIR_NAMES = os.listdir(TRAIN_PATH)
count=1
X_Axis_TR=[]
Y_Axis_TR=[]
for sub_folder_name in TRAIN_DIR_NAMES:
 TRAIN_FILE_NAMES = os.listdir(TRAIN_PATH+"\\"+sub_folder_name)
 X_Axis_TR.append(sub_folder_name)
 Y_Axis_TR.append(len(TRAIN_FILE_NAMES))
plt.plot(X_Axis_TR, Y_Axis_TR,label="Train Images")
plt.legend()
plt.show()      
 