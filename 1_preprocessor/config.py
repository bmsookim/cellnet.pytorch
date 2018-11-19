############# Configuration file #############

# Name of dataset
#name = 'Granulocytes_vs_Mononuclear'
#name = 'Granulocytes'
#name = 'Mononuclear'
#name = 'Lymphocyte'
#name = 'WBC'

# Open dataset testout
#name = 'BCCD'
#name = 'AUG_BCCD'
#name = 'GM_BCCD'
#name = 'G_BCCD'
#name = 'M_BCCD'
#name = 'AUG_GM'
#name = 'AUG_G'
name = 'AUG_M'

# Base directory for data formats
data_base = '/mnt/datasets/' + name

# Base directory for augmented data formats
resize_base = '/home/bumsoo/Data/resized/'
split_base = '/home/bumsoo/Data/_train_val/'

# Directory for data formats
resize_dir = resize_base + name
split_dir = split_base + name

# Train augmentation
rotate_mode = 'strict'

# Validation split
split = 'ratio' # [ratio/fix]
val_ratio = 0.2
val_num = 25
