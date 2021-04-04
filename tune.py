
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
from keras.callbacks import Callback
from keras.optimizers import Adam as Adam

import os
import sys
import zipfile
import math
import train



from keras import backend as K
from keras.utils.generic_utils import CustomObjectScope
import shutil

zip_ref = zipfile.ZipFile(sys.argv[1], 'r')
zip_ref.extractall("./test/train")
zip_ref.close()

zip_ref = zipfile.ZipFile(sys.argv[2], 'r')
zip_ref.extractall("./test/validation")
zip_ref.close()

#str = sys.argv[2]
#file = open(str[2:],'r')


img_width, img_height = 224,224 #image resolution hyper parameter
train_data_dir = "./test/train" #link of train data directory
validation_data_dir = "./test/validation" #link of validation data directory

img_channels = 3 #RGB
nb_classes = 8 #final categories
batch_size = 8 #how many images at a time, can be treated as a Hyper Parameter
nb_epoch = 10 #iterations


#how many batches per epoch

hyper_alpha = 0.0
hyper_img_width = 0.0

i=1
j=1
hyper_score = -9999

tune_file = open(sys.argv[3],"w")


for alpha_1 in [1.0,.75,.50,.25]:
    for img_width in [224,192,160,128]:
        training_set, number_of_train_images = train.training_data(train_data_dir, img_width, img_width, batch_size)

        validation_set, number_of_test_images = train.validate_data(validation_data_dir, img_width, img_width,
                                                                    batch_size)

        steps_per_epoch = math.ceil(number_of_train_images / batch_size)
        validation_steps = math.ceil(number_of_test_images / batch_size)

        model_final = train.model_creator(alpha_1,img_width,img_width,img_channels,nb_classes)
        history = model_final.fit_generator(training_set,
                            steps_per_epoch = steps_per_epoch,
                            epochs=nb_epoch,
                            validation_data=validation_set,
                            validation_steps = validation_steps,
                            initial_epoch=0)

        acc = history.history['accuracy'][-1]
        loss = history.history['loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]



        print ("alpha : "+str(alpha_1)+" img_width: "+str(img_width)+" accuracy: "+ str(val_acc)+"\n")

        tune_file.write("alpha : "+str(alpha_1)+" img_width: "+str(img_width)+" tr_acc: "+ str(acc)+ " tr_loss: "+ str(loss)+" val_acc: "+ str(val_acc)+" val_loss: "+ str(val_loss)+ "\n")

        if(hyper_score < val_acc):
            hyper_score = val_acc
            hyper_alpha = alpha_1
            hyper_img_width = img_width

    i+=1

file_1 = open(sys.argv[4],"w")
file_1.write(str(hyper_alpha)+"\n")
file_1.write(str(hyper_img_width))

shutil.rmtree("./test")


