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
mean = [0.76937065622596712, 0.62102846073902673, 0.62280950464590923]
std = [0.20634491876598526, 0.27042467744347987, 0.24373346183373229]
