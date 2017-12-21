############# Configuration file #############

# Name of dataset
name = 'CELL_PATCHES'

# Base directory for data formats
data_base = '/mnt/datasets/' + name

# Base directory for augmented data formats
resize_base = '/home/bumsoo/Data/resized/'
split_base = '/home/bumsoo/Data/split/'

# Directory for data formats
resize_dir = resize_base + name
split_dir = split_base + name

# Validation split
split = 'fix' # [ratio/fix]
val_ratio = 0.2
val_num = 10
