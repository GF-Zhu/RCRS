# -*- encoding: utf-8 -*-

from solar_center_radius_model import *
from solar_data import *
import time
from keras.callbacks import Callback

start = time.clock()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_size = 8
num_of_all_image = 93686 #the total number of dataset
prop_training = 0.8  # the data split proportion (for training)
num_epochs = 50

num_steps_per_train_epoch = np.floor(num_of_all_image * prop_training / batch_size)
num_steps_per_val_epoch = np.floor(num_of_all_image * (1 - prop_training) / batch_size)
train_dir = './data/train' # the original training images
labels_dir = './data/data.txt' #the center coordinates and radius of the solar disk
is_binary_label = False

#to save loss and r_square
class LossHistory(Callback):
    def on_epoch_end(self, batch, logs={}):
        try:
            f = open("log/epoch_history.txt", 'a')
            f.write(str(logs.get('loss')) + " " +str(logs.get('r_square')) + " " + str(logs.get('val_loss')) + " " +str(logs.get('val_r_square')))
            f.write("\n")
            f.close()
        except:
            pass

#the training generator
trainGene = my_dataset_generator(batch_size,
                         train_dir=train_dir,
                         labels_dir=labels_dir,
                         train_set='train',
                         training_rate = prop_training,
                         is_binary_label=is_binary_label)
#the validation generator
valGene = my_dataset_generator(batch_size,
                         train_dir=train_dir,
                         labels_dir=labels_dir,
                         train_set='val',
                         training_rate = prop_training,
                         is_binary_label=is_binary_label)
#the testing generator
testGene = my_dataset_generator(batch_size,
                         train_dir=train_dir,
                         labels_dir=labels_dir,
                         train_set='test',
                         training_rate = prop_training,
                         is_binary_label=is_binary_label)
model = solar_center_radius_cnn(is_binary_label=is_binary_label)

lrScheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_delta=1e-4, min_lr=1e-8)  # the learning rate decay
model_checkpoint = ModelCheckpoint('solar_center_radius_model.hdf5', monitor='loss',verbose=1, save_best_only=True,save_weights_only=True)

history = LossHistory()
model.fit_generator(trainGene,steps_per_epoch=num_steps_per_train_epoch,epochs=num_epochs,callbacks=[model_checkpoint,lrScheduler,history], validation_data=valGene, validation_steps=num_steps_per_val_epoch,verbose=2)
#model.summary()

score = model.evaluate_generator(testGene, steps=num_steps_per_val_epoch, verbose=1)
print('Test loss:', score[0])
print('Test r2:', score[1])
print('Test lr:', score[2])

end=time.clock()
print('Running time: %s Seconds'%(end-start))




