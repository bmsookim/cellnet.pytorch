# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/cellnet.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Grad CAM Implementation
#
# Description : grad_cam.py
# The main code for implementations of grad-CAM.
# ***********************************************************

from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import functional as F

class PropagationBase(object):
    def __init__(self, model, cuda=False): # PropagationBase(model, use_gpu)
        self.model = model
        self.model.eval()

        if cuda:
            self.model.cuda()

        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func() # ?
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot
