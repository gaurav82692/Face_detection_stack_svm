import datetime

def log(text):
    f=open("logs_dataset_preparation.txt","a")
    f.write(text+" :: "+str(datetime.datetime.now()) +"\n")
    print(text+" :: "+str(datetime.datetime.now()) +"\n")
    f.close()

if __name__ == "__main__":
    print ("This file can be executed from main only")