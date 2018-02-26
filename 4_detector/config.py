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
mean = [0.76068656077103536, 0.58102377925859161, 0.59144916648061485]
std = [0.20279549420263882, 0.28521133531479254, 0.25443677082110039]
