#################### Configuration File ####################

# Base directory for data formats
#name = 'Granulocytes_vs_Mononuclear'
name = 'BCCD'

H1_name = 'Granulocytes_vs_Mononuclear'
G_name = 'Granulocytes'
M_name = 'Mononuclear'
N_name = 'Neutrophil'
L_name = 'Lymphocyte'

data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/split/'

# Databases for each formats
aug_base = '/home/bumsoo/Data/split/' + name
test_base = '/home/bumsoo/Data/test/WBC_TEST'

# model option
batch_size = 16
num_epochs = 70
lr_decay_epoch=35
momentum = 0
feature_size = 500

# Granulocyte vs Mononuclear
mean_H = [0.75086572277254926, 0.54344990735699861, 0.56189840210810549]
std_H = [0.19795568869316291, 0.29897863665208158, 0.26473830163404605]

# Granulocytes
mean_G = [0.7260078523435356, 0.50995708667892348, 0.53427415115119681]
std_G = [0.20467739828829676, 0.30475251036479978, 0.27280720957235177]

# Mononuclear
mean_M = [0.7594375520640767, 0.55067103657252647, 0.56366851380109106]
std_M = [0.19572272727963994, 0.30284308059727105, 0.26669885653951991]

# NH meanstd
mean_N = [0.72968820508788612, 0.52224113128933247, 0.54099372274735391]
std_N = [0.208528564775461, 0.30056530735626585, 0.27138967466099473]

# LH meanstd
mean_L = [0.7571956979879545, 0.55694333649406613, 0.56854173074367431]
std_L = [0.20890086199641186, 0.31668580372231542, 0.28084878897340337]

# Overall meanstd
#mean = [0.76937065622596712, 0.62102846073902673, 0.62280950464590923]
#std = [0.20634491876598526, 0.27042467744347987, 0.24373346183373229]

# BCCD meanstd
mean = [0.66049439066232607, 0.64131680516457479, 0.67861641316853616]
std = [0.25870889538041947, 0.26112642565510558, 0.26200774691285844]
