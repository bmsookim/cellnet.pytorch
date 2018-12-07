#################### Configuration File ####################

# Base directory for data formats
#name = 'GM_BCCD'
#name = 'AUG_BCCD'
name = 'AUG_GM'

data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/_train_val/'

# Databases for each formats
data_base = data_dir + name
aug_base = '/home/bumsoo/Data/_train_val/' + name
test_base = '/home/bumsoo/Data/_test/BCCD_FULL/'
test_dir = '/home/bumsoo/Data/_test/' # Number on the back


# model option
batch_size = 16
num_epochs = 70
lr_decay_epoch=35
momentum = 0
feature_size = 500
mode = 'aug'

if mode == 'original':
    # Granulocyte vs Mononuclear
    mean = [0.7734852310658247, 0.6001978670527376, 0.6681920291069585]
    std = [0.05499360963837181, 0.15174074470693394, 0.10608604828874389]
elif (name == 'AUG_BCCD' or name == 'AUG_GM'):
    mean = [0.66049439066232607, 0.64131680516457479, 0.67861641316853616]
    std = [0.25870889538041947, 0.26112642565510558, 0.26200774691285844]
elif (name == 'BCCD'):
    mean = [0.7734852310658247, 0.6001978670527376, 0.6681920291069585]
    std = [0.05499360963837181, 0.15174074470693394, 0.10608604828874389]
elif (name == 'GM_BCCD'):
    mean = [0.7734852310658247, 0.6001978670527376, 0.6681920291069585]
    std = [0.05499360963837181, 0.15174074470693394, 0.10608604828874389]
else:
    raise NotImplementedError
