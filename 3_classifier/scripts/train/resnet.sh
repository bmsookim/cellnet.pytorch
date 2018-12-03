python3 main.py \
    --optimizer SGD \
    --lr 0.0045 \
    --lr_decay 0.94 \
    --lr_decay_epoch 2 \
    --weight_decay 5e-4 \
    --net_type resnet \
    --depth 50 \
    --resetClassifier \
    --finetune
    #--testOnly
