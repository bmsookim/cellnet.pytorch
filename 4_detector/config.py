#################### Configuration File ####################

# Base directory for data formats
name = 'Granulocytes_vs_Mononuclear'
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

"""
# Global meanstd
#mean = [0.76937065622596712, 0.62102846073902673, 0.62280950464590923]
#std = [0.20634491876598526, 0.27042467744347987, 0.24373346183373229]

# H1 meanstd (NH)
#h1_mean = [0.72968820508788612, 0.52224113128933247, 0.54099372274735391]
#h1_std = [0.208528564775461, 0.30056530735626585, 0.27138967466099473]

# H2 meanstd (LH)
#h2_mean = [0.7571956979879545, 0.55694333649406613, 0.56854173074367431]
#h2_std = [0.20890086199641186, 0.31668580372231542, 0.28084878897340337]
#h2_mean = [0.75086572277254915, 0.54344990735699861, 0.56189840210810549]
#h2_std = [0.19795568869316291, 0.2989786366520818, 0.26473830163404605]

# WBC_Only
#mean = [0.75086572277254915, 0.54344990735699861, 0.56189840210810549]
#std = [0.19795568869316291, 0.2989786366520818, 0.26473830163404605]
"""
