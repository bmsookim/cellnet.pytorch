#################### Configuration File ####################

# Base directory for data formats
name = 'Granulocytes_vs_Mononuclear'
#name = 'BC_TRAIN' # No hierarchy

data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'
test_dir = '/home/bumsoo/Data/test/'

data_base = data_dir + name
aug_base = aug_dir + name
test_base = test_dir + name

# model directory
model_dir = '../3_classifier/checkpoints'
#test_dir = '/home/bumsoo/Data/test/BC_TEST/TEST' # Number on the back
test_img_dir = test_dir+'/BC_TEST_img'

# model option
batch_size = 16
num_epochs = 100
lr_decay_epoch=20
feature_size = 500

# Global meanstd
mean = [0.75086572277254926, 0.54344990735699861, 0.56189840210810549]
std = [0.19795568869316291, 0.29897863665208158, 0.26473830163404605]

# MICCAI_train meanstd
#mean = [0.76937065622596712, 0.62102846073902673, 0.62280950464590923]
#std = [0.20634491876598526, 0.27042467744347987, 0.24373346183373229]
