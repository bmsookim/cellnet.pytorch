#################### Configuration File ####################

# Base directory for data formats
name = 'WBC_CAD'
data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'
test_dir = '/home/bumsoo/Data/test/'

data_base = data_dir + name
aug_base = aug_dir + name
test_base = test_dir + name

# model directory
model_dir = '../3_classifier/checkpoints'
test_dir = '/home/bumsoo/Data/test/20_CELL_TEST/TEST' # Number on the back

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.75625414776226185, 0.57571597431337018, 0.58672883441364954]
std = [0.20549549425609484, 0.28728030921184039, 0.25679252267543851]
