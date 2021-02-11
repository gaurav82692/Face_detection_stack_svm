import datetime

def log(text):
    f=open("C:/Users/Gaurav/Desktop/DIP_Project/Data Set Prepration/logs_my_model.txt","a")
    f.write(text+" :: "+str(datetime.datetime.now()) +"\n")
    print(text+" :: "+str(datetime.datetime.now()) +"\n")
    f.close()

if __name__ == "__main__":
    print ("This file can be executed from main only")