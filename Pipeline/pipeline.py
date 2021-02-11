'''
Requirement: This file needs the data to be preprocessed.
It'll require the images folder, make sure you are running this script on the cleaned dataset. 
You may alter the folder paths to save them acording to your machiene's directory structure.
Modify the num_classes attribute value to alter the number of celebrities to be choosen.
'''

from train_test_split import train_test_split
from image_corrector import correct
from sharpen import sharpen
import threading, queue

q = queue.Queue()

def worker():
    global q
    while True:
        item = q.get()
        q.task_done()

def pipeline():
    global q
    # turn-on the worker thread
    threading.Thread(target=worker, daemon=True).start()

    q.put(correct())
    q.put(sharpen())
    q.put(train_test_split())
    q.join()

if __name__ == '__main__':
    pipeline()
    
