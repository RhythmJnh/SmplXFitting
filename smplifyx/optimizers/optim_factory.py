# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.optim as optim
from .lbfgs_ls import LBFGS as LBFGSLs


def create_optimizer(parameters, optim_type='lbfgsls',
                     lr=1.0,
                     momentum=0.9,
                     use_nesterov=True,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-8,
                     use_locking=False,
                     weight_decay=0.0,
                     centered=False,
                     rmsprop_alpha=0.99,
                     maxiters=30,
                     gtol=1e-9,
                     ftol=1e-9,
                     **kwargs):
    ''' Creates the optimizer
    '''
    if optim_type == 'lbfgsls':
        return (LBFGSLs(parameters, lr=lr, max_iter=maxiters,
                       line_search_fn='strong_Wolfe'), False)
    else:
        raise ValueError('Optimizer {} not supported!'.format(optim_type))