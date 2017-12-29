#################### Configuration File ####################

# Base directory for data formats
name = 'CELL_PATCHES'
test_dir = '/home/bumsoo/Data/test/cell_test'

data_base = '/home/mnt/datasets/'+name
aug_base = '/home/bumsoo/Data/split/'+name

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.778163803690477, 0.62366406713856704, 0.62748488269742386]
std = [0.18654390350275066, 0.25577185166630317, 0.22957029180170951]
