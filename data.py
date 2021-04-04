import zipfile
import os
import glob
from PIL import Image
import shutil
import sys
zip_data_dir = sys.argv[1]
traindir1 = sys.argv[2]
validate = sys.argv[3]
test = sys.argv[4]
traindir2 = sys.argv[5]
traindir3 = sys.argv[6]
simple3 = sys.argv[7]
def func(str):
    for i in os.listdir(str):
        if os.path.isdir(os.path.join(str, i)):
            shutil.rmtree(os.path.join(str, i))


func("./Data")

print("clear Data folder.......")

print("start unzip data.zip.......")
zip_ref = zipfile.ZipFile(sys.argv[1], 'r')
zip_ref.extractall("./Data/temp")
zip_ref.close()

print("end unzip data.zip............")

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))

def zipit(str,dir_list, zip_name):
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for diri in dir_list:
        zipdir(str+diri, zipf)
    zipf.close()

def saveimage(file,str1,text,counter):
    new_im = Image.open(file)
    if not os.path.exists(str1+text+"/"):
        os.makedirs(str1+text+"/")
    new_im.save(str1 +text+"/" + str(counter) + ".jpeg")
    new_im.close()


for text in os.listdir("./Data/temp"):
    jpeg_file_path = os.path.join("./Data/temp", text + '/*')
    files = glob.glob(jpeg_file_path)
    validatecount=int(0.1*len(files))
    testcount=int(0.1*len(files))
    traincount=len(files)-validatecount - testcount
    counter=1
    counter1 = 0
    print("working with........ "  + str(text))
    for file in files:
        if(counter<=validatecount):
           if(counter<=3):
               saveimage(file,simple3,text,counter)

           saveimage(file,validate,text,counter)
        elif(counter<=validatecount + testcount):
            saveimage(file,test,text,counter)
        else:
            saveimage(file,traindir1,text,counter)
            if counter1 <= traincount*0.05:
                counter1+=1
                saveimage(file,traindir2,text,counter)
                saveimage(file,traindir3,text,counter)

            elif counter1 <= traincount*0.2:
                counter1+=1
                saveimage(file,traindir3,text,counter)
        counter=counter+1

print("creating zip.............")

zipit(test,os.listdir(test),test+"data.zip")
zipit(validate,os.listdir(validate),validate+"data.zip")
zipit(traindir1,os.listdir(traindir1),traindir1+"data.zip")
zipit(traindir2,os.listdir(traindir2),traindir2+"data.zip")
zipit(traindir3,os.listdir(traindir3),traindir3+"data.zip")
zipit(simple3,os.listdir(simple3),simple3+"data.zip")


print("clear files.........")

func(test)
func(validate)
func(traindir1)
func(traindir2)
func(traindir3)
func(simple3)

shutil.rmtree("./Data/temp")


