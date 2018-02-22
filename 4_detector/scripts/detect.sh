#!/bin/bash

# subtypes = ['WBC_Basophil', 'WBC_Eosinophil', 'WBC_Lymphocyte', 'WBC_Lymphocyte_atypical', 'WBC_Metamyelocyte', 'WBC_Monocyte', 'WBC_Myelocyte', 'WBC_Neutrophil_Band', 'WBC_Neutrophil_Segmented', 'WBC_Smudge']

for ((i=6; i<=21; i++)) do
    python detect_cell.py \
        --depth 50 \
        --windowSize 100 \
        --stepSize 50 \
        --testNumber $i \
        --subtype RBC_Crop
done
