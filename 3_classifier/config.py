#################### Configuration File ####################

# Base directory for data formats
#name = 'Granulocytes_vs_Mononuclear'
#name = 'BCCD'
#name = 'AUG_BCCD'
#name = 'WBC'
#name = 'GM_BCCD'
#name = 'G_BCCD'
#name = 'M_BCCD'
name = 'AUG_GM'
#name = 'AUG_G'
#name = 'AUG_M'

H1_name = 'Granulocytes_vs_Mononuclear'
G_name = 'Granulocytes'
M_name = 'Mononuclear'
N_name = 'Neutrophil'
L_name = 'Lymphocyte'

data_dir = '/home/mnt/datasets/'
aug_dir = '/home/bumsoo/Data/_train_val/'

# Databases for each formats
aug_base = '/home/bumsoo/Data/_train_val/' + name
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

if (name == 'Granulocytes_vs_Mononuclear'):
    # Granulocytes_vs_Mononuclear
    mean = [0.75086572277254926, 0.54344990735699861, 0.56189840210810549]
    std = [0.19795568869316291, 0.29897863665208158, 0.26473830163404605]
    print(name)
    print(mean)
    print(std)
elif (name == 'WBC'):
    # WBC meanstd
    mean = [0.7593608074350131, 0.6122998654014106, 0.6142165029355519]
    std = [0.22106204895546486, 0.27805751343124707, 0.2522135438853085]
    print(name)
    print(mean)
    print(std)
elif (name == 'AUG_BCCD' or name == 'AUG_GM'):
    mean = [0.66049439066232607, 0.64131680516457479, 0.67861641316853616]
    std = [0.25870889538041947, 0.26112642565510558, 0.26200774691285844]
elif (name == 'BCCD' or name == 'GM_BCCD'):
    mean = [0.7734852310658247, 0.6001978670527376, 0.6681920291069585]
    std = [0.05499360963837181, 0.15174074470693394, 0.10608604828874389]
elif (name == 'G_BCCD'):
    mean = [0.7718285852001573, 0.6022119763509954, 0.6749639709800079]
    std = [0.056041130364610635, 0.14721271026408136, 0.10112752896514536]
elif (name == 'M_BCCD'):
    mean = [0.7808339422135298, 0.5912634847809788, 0.6381523895162508]
    std = [0.050083456200428145, 0.17038192897611737, 0.1257458245396846]
elif (name == 'AUG_G'):
    mean = [0.6573611811732865, 0.6380462082799403, 0.6767932371428185]
    std = [0.25759305343128985, 0.2599911268172555, 0.26104541183046104]
elif (name == 'AUG_M'):
    mean = [0.6636497050358816, 0.644610476205828, 0.6804524517544022]
    std = [0.2598277668079999, 0.2622647674637204, 0.26297331235868215]
else:
    raise NotImplementedError
