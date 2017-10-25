import caffe
from caffe import layers as L, params as P
import scipy.io as sio
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import h5py


## ================= PREDEFINE NETWORK LAYER ==============================
# def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
#     ''' Define convolution and relu layer together with default setting '''
#     conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
#                          num_output=nout, pad=pad, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
#                          param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
#     return conv, L.ReLU(conv, in_place=True)

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, filter_type = 'gaussian', lr_mult1 = 1):
    ''' Define convolution and relu layer together with default setting '''
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, weight_filler=dict(type= filter_type, std = 0.1), bias_filler=dict(type='constant', value = 0),
                         param=[dict(lr_mult= lr_mult1), dict(lr_mult= 0.1)])
    return conv, L.ReLU(conv, in_place=True)
def convLayer(bottom, nout, ks = 3, stride = 1, pad = 1, filter_type= 'gaussian', lr_mult1 = 1):
    ''' Define convolution layer with default setting '''
    return L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, weight_filler=dict(type= filter_type, std = 0.1), bias_filler=dict(type='constant', value = 0),
                         param=[dict(lr_mult= lr_mult1), dict(lr_mult=0.1)])
def conv_bn_relu(bottom, nout, ks=3, stride = 1, pad = 1, std_in = 0.001):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, 
                        weight_filler=dict(type='gaussian', std = std_in), bias_term = False,
                        param = [dict(lr_mult = 1)])
    batn = L.BatchNorm(conv)
    scale = L.Scale(batn, scale_param = dict(bias_term = True))
    relu = L.ReLU(scale)

    return conv, batn, scale, relu
def batNormLayer(bottom):
    batnorm = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=False), in_place=False,
                              param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),
                                     dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batnorm, bias_term=True, in_place=True,filler=dict(value=1), bias_filler=dict(value=0))    
    return batnorm, scale
def gen_net(split, batch_size, blk_size, subrate, source_path = ''):
    ''' Define network '''
    n = caffe.NetSpec()
    #n.name = net_type; 

    # ====================== Step 1. Define the input layer ==========================================
    if net_type == 'ReconNet' or net_type == 'DR2_Stage1' or net_type == 'DR2_Stage2':
        noMeas = int (round(subrate * blk_size * blk_size))
        if split == 'train' :
            n.data, n.label = L.HDF5Data(name="data", batch_size = batch_size, source=source_path, ntop=2,
                                        include={'phase': caffe.TRAIN}, type = "HDF5Data")
        elif split == 'test' :
            n.data, n.label = L.HDF5Data(name="data", batch_size = 2, source=source_path, ntop=2,
                                        include={'phase': caffe.TEST}, type = "HDF5Data")
        else:
            n.data = L.Input(name="data", ntop=1, input_param={'shape': {'dim': [1, noMeas, 1, 1]}})       
    
    elif net_type == 'CSNet':
        if split == 'train_val' :
            n.data, n.label = L.HDF5Data(name="data", batch_size = batch_size, source=source_path, ntop=2,
                                        include={'phase': caffe.TRAIN}, type = "HDF5Data")
            n.data, n.label = L.HDF5Data(name="data", batch_size = 2, source=source_path, ntop=2, 
                                        include={'phase': caffe.TEST}, type = "HDF5Data")
        else:
            n.data = L.Input(name="data", ntop=1, input_param={'shape': {'dim': [1, 1, blk_size, blk_size]}})
        
    # ======================= Stage 2: Define the main network =======================================
    if net_type == 'ReconNet' or net_type == 'DR2_Stage1' or net_type == 'DR2_Stage2':
        
        if subrate == 0.01:
            adap_std = 0.01
        elif subrate == 0.04:
            adap_std = 0.03
        elif subrate == 0.25:
            adap_std = 0.05
        else:
            adap_std = 0.05

        n.fc1 = L.Convolution(n.data, kernel_size=1, stride=1, num_output=blk_size*blk_size, pad=0, 
                        weight_filler=dict(type='gaussian', std = adap_std), bias_filler=dict(type='constant', value = 0),
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
        n.reshape = L.Reshape(n.fc1, reshape_param = {'shape' : {'dim': [0, 1, blk_size, blk_size]}})

        if net_type == 'ReconNet':            
            n.conv1, n.relu1 = conv_relu(n.reshape, 64, 11, 1, 5)    
            n.conv2, n.relu2 = conv_relu(n.relu1, 32, 1, 1, 0)    
            n.conv3, n.relu3 = conv_relu(n.relu2, 1,  7, 1, 3, 'gaussian', 0.1)    
            n.conv4, n.relu4 = conv_relu(n.relu3, 64, 11, 1, 5)    
            n.conv5, n.relu5 = conv_relu(n.relu4, 32, 1,  1, 0)
            n.conv6 = convLayer(n.relu5, 1, 7,  1, 3, 'gaussian', 0.1)

            if (split == 'train' or  split == 'test'):
                n.loss = L.EuclideanLoss(n.conv6, n.label)	

        elif net_type == 'DR2_Stage1':
            if (split == 'train' or  split == 'test'):
                n.loss = L.EuclideanLoss(n.reshape, n.label, loss_weight = 1)	

        elif net_type == 'DR2_Stage2':            
            # 1st residual subnet
            n.conv1r, n.bnorm1, n.scale1, n.relu1 = conv_bn_relu(n.reshape, 64, 11, 1, 5)
            n.conv2r, n.bnorm2, n.scale2, n.relu2 = conv_bn_relu(n.relu1, 32, 1, 1, 0)
            n.conv3r =  L.Convolution(n.relu2, kernel_size = 7, stride = 1, num_output = 1, pad = 3, 
                        weight_filler = dict(type='gaussian', std = 0.001), bias_term = False,
                        param=[dict(lr_mult= 0.1)])
            n.res1 = L.Eltwise(n.reshape, n.conv3r)
            
            # 2nd Residual subnet
            n.conv4r, n.bnorm4, n.scale4, n.relu4 = conv_bn_relu(n.res1, 64, 11, 1, 5)
            n.conv5r, n.bnorm5, n.scale5, n.relu5 = conv_bn_relu(n.relu4, 32, 1, 1, 0)
            n.conv6r =  L.Convolution(n.relu5, kernel_size = 7, stride = 1, num_output = 1, pad = 3, 
                        weight_filler = dict(type='gaussian', std = 0.001), bias_term = False,
                        param=[dict(lr_mult= 0.1)])
            n.res2 = L.Eltwise(n.res1, n.conv6r)

            # 3rd Residual subnet
            n.conv7r, n.bnorm7, n.scale7, n.relu7 = conv_bn_relu(n.res2, 64, 11, 1, 5)
            n.conv8r, n.bnorm8, n.scale8, n.relu8 = conv_bn_relu(n.relu7, 32, 1, 1, 0)
            n.conv9r =  L.Convolution(n.relu8, kernel_size = 7, stride = 1, num_output = 1, pad = 3, 
                        weight_filler = dict(type='gaussian', std = 0.001), bias_term = False,
                        param=[dict(lr_mult= 0.1)])
            n.res3 = L.Eltwise(n.res2, n.conv9r)

            # 4th Residual subnet
            n.conv10r, n.bnorm10, n.scale10, n.relu10 = conv_bn_relu(n.res3, 64, 11, 1, 5)
            n.conv11r, n.bnorm11, n.scale11, n.relu11 = conv_bn_relu(n.relu10, 32, 1, 1, 0)
            n.conv12r =  L.Convolution(n.relu11, kernel_size = 7, stride = 1, num_output = 1, pad = 3, 
                        weight_filler = dict(type='gaussian', std = 0.001), bias_term = False,
                        param=[dict(lr_mult= 0.1)])
            n.res4 = L.Eltwise(n.res3, n.conv12r)

            # Loss layer 
            if (split == 'train' or  split == 'test'):
                n.loss = L.EuclideanLoss(n.res4, n.label, loss_weight = 1)	                
                n.loss2 = L.EuclideanLoss(n.reshape, n.label, loss_weight = 0)	

       
        return n.to_proto()

## ================= CONSTRUCT THE NETWORK PROTOTXT ==============================
def make_net(net_type, net_mode, batch_size, blk_size, subrate, source_path, note):
    ''' Making particlar network for each data of : train, test or deploy'''
    # if net_type == 'DR2_Stage1' or net_type == 'DR2_Stage2':
    #     folder = 'solvers/DR2'
    # else:
    folder = 'solvers/' + net_type + '/subrate_0_' + note

    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + '\\' + net_mode + '_' + net_type + '.prototxt', 'w') as f:
        net_prototxt = str(gen_net(net_mode, batch_size, blk_size, subrate, source_path ))
        f.write(net_prototxt)

def make_all_net(net_type, batch_size, blk_size, subrate, train_path, test_path, note):
    ''' Making networks: train, test and deploy '''
    make_net(net_type, 'train', batch_size, blk_size, subrate, train_path, note)
    make_net(net_type, 'test', batch_size, blk_size, subrate, test_path, note)
    make_net(net_type, 'deploy', batch_size, blk_size, subrate, '', note)

## ================= CONSTRUCT THE SOLVER PROTOTXT ==============================
def make_solver(net_type, subrate, solver_type, lr_rate, max_iter, momentum, note):
    s = caffe_pb2.SolverParameter()

    # Set a seed for reproducible experiments: controls randomization in training.
    s.random_seed = 0xCAFFE

    # Specify locations of the train and (maybe) test networks.
    s.train_net =  'train_' + net_type + '.prototxt'
    s.test_net.append( 'test_' + net_type + '.prototxt')

    s.test_interval = 1000  # Test after every 500 training iterations.
    s.test_iter.append(1000)  # Test on 100 batches each time we test.

    s.max_iter = max_iter  # no. of times to update the net (training iterations)

    # solver types include "SGD", "Adam", and "Nesterov" among others.
    s.type = solver_type

    # Set the initial learning rate, real value
    s.base_lr = lr_rate

    # How much the previous weight will be retained in the new calculations.
    s.momentum = momentum

    # Set weight decay to regularize and prevent overfitting for large weight
    s.weight_decay = 1e-4

    # Set `lr_policy` to define how the learning rate changes during training.
    # This is the same policy as our default LeNet.
    s.lr_policy = 'fixed'

    # define how much the learning rate should change every time we reach the next "step."
    s.gamma = 0.5

    # Display the current training loss and accuracy every 100 iterations.
    s.display = 500

    # Snapshots are files used to store networks we've trained.
    # We'll snapshot every 5K iterations -- twice during training.
    s.snapshot = 1000
   
    s.snapshot_prefix = 'snapshot/' + net_type + '_0_' + note

    s.average_loss = 20
    s.iter_size = 1
    s.test_initialization = False

    # if net_type == 'DR2_Stage1' or net_type == 'DR2_Stage2':
    #     folder = 'solvers/DR2'
    # else:
    #     folder = 'solvers/' + net_type
    folder = 'solvers/' + net_type + '/subrate_0_' + note 
    if not os.path.exists(folder + '/snapshot'):
        os.makedirs(folder + '/snapshot')

    # Caffe mode GPU or CPU
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    solver_path = folder + '/solver_' + net_type + '.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(s))
    return s, solver_path

def make_train_sh(net_type, subrate, gpu_id, note):
    ''' Making train.sh '''
    folder = 'solvers/' + net_type + '/subrate_0_'  + note 
    if not os.path.exists(folder ):
        os.makedirs(folder)
    train_cmd_path = folder + '/train_' + net_type + '.sh'    

    with open(train_cmd_path, 'w') as f:
        f.write('rm ' + net_type + '.log' + '\n' )     # the first file
        if net_type == 'DR2_Stage2':
            weight_loc = '../../DR2_Stage1/subrate_0_' + note  + '/snapshot/DR2_Stage1_0_' + note + '_iter_100000.caffemodel'  ;
            train_str = '/mnt/e/Research_DLCS/caffe-windows-bvlc/build/tools/Release/caffe.exe train --solver=solver_' + net_type + '.prototxt --weights=' + weight_loc + ' --gpu ' + str(gpu_id) + '|  tee -a ' + net_type + '.log'            
        else:
            train_str = '/mnt/e/Research_DLCS/caffe-windows-bvlc/build/tools/Release/caffe.exe train --solver=solver_' + net_type + '.prototxt --gpu ' + str(gpu_id) + '|  tee -a ' + net_type + '.log'
        f.write(train_str)

# ================================= MAIN FUNCTION ==============================
if __name__ == '__main__':
    # Basic information
    batch_size = 128
    blk_size = 33   
    net_type = 'DR2_Stage2'
    subrate = 0.25   

    if subrate < 0.1:
        note = '0' + ('{:.0f}').format(subrate * 100)
    else:
        note = ('{:.0f}').format(subrate * 100)

    gpu_id = 1; 

    # ------------ For data generation -------------------   
    if net_type == 'DR2_Stage1' or net_type == 'DR2_Stage2' or net_type == 'ReconNet':
        train_path = '../../../data/CS_Meas/hdf5_org/Train_r' + str(subrate) + '_' + \
                    str(blk_size * blk_size) +'.txt'
        test_path = '../../../data/CS_Meas/hdf5_org/Test_r' + str(subrate) + '_' +\
                    str(blk_size * blk_size) +'.txt'
    
    # ------------ Making prototxt network ------------------
    print "Step 2. Making network!"
    make_all_net(net_type, batch_size, blk_size, subrate, train_path, test_path, note)

    # making solver
    solver_type = 'Adam'
    if net_type == 'DR2_Stage1':
        lr_rate = 1e-2
    else:
        lr_rate = 1e-5

    max_iter = 100000
    momentum = 0.9
    print "Step 3. Making solver!"
    s, solver_path = make_solver(net_type, subrate, solver_type, lr_rate, max_iter, momentum, note)
    
    make_train_sh(net_type, subrate, gpu_id, note)

    print "Generation Netework End !"