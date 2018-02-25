#!/bin/bash

# subtypes = ['WBC_Basophil', 'WBC_Eosinophil', 'WBC_Lymphocyte', 'WBC_Lymphocyte_atypical', 'WBC_Monocyte', 'WBC_Neutrophil_Band', 'WBC_Neutrophil_Segmented', 'WBC_Smudge']

for ((i=1; i<=21; i++)) do
    python detect_cell.py \
        --depth 50 \
        --windowSize 80 \
        --stepSize 60 \
        --testNumber $i
        #--subtype WBC_Lymphocyte
done
