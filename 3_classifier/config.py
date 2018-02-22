#################### Configuration File ####################

# Base directory for data formats
name = 'FIND_WBC'
data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'
test_dir = '/home/bumsoo/Data/test/'

data_base = data_dir + name
aug_base = aug_dir + name
test_base = test_dir + name

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.80281052043887768, 0.67348388118712643, 0.67175728590459316]
std = [0.18278017174219421, 0.24164077214496718, 0.21683084040751013]
