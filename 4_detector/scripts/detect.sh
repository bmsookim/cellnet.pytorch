#!/bin/bash

# subtypes = ['WBC_Basophil', 'WBC_Eosinophil', 'WBC_Lymphocyte', 'WBC_Lymphocyte_atypical', 'WBC_Monocyte', 'WBC_Neutrophil_Band', 'WBC_Neutrophil_Segmented', 'WBC_Smudge']

for ((i=2; i<=2; i++)) do
    python detect_cell.py \
        --depth 50 \
        --windowSize 70 \
        --stepSize 50 \
        --testNumber $i
        #--subtype WBC_Lymphocyte
done
