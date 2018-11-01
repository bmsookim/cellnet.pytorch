#################### Configuration File ####################

# Base directory for data formats
#name = 'Granulocytes_vs_Mononuclear'
name = 'WBC' # No hierarchy

data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/_train_val/'

data_base = data_dir + name
aug_base = aug_dir + name

# model directory
model_dir = '../3_classifier/checkpoints'
test_dir = '/home/bumsoo/Data/_test/Cell_Detect/' # Number on the back

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# WBC meanstd
mean = [0.7593608074350131, 0.6122998654014106, 0.6142165029355519]
std = [0.22106204895546486, 0.27805751343124707, 0.2522135438853085]
