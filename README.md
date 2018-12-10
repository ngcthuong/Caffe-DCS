# Deep Compressive Sensing in Caffe
Collection of source code fore deep learning based compressive sensing in Caffe. 

### Disclaimer 
All tools are actually reimplemented with some improvements as
1. For ReconNet and DR2Net: 
- Implemented as fully-convolutional network
- Generated files with pycaffe (train, test, solver - prototxt as well as train.sh).
- Data generation is added: sampling matrices, separate training data for different subrate, more test data, etc. 
- General test framework: applied for both ReconNet and DR2Net

### Collection Tools
1. ReconNet https://github.com/KuldeepKulkarni/ReconNet
2. DR2Net https://github.com/AtenaKid/caffe_dr2

### References
[1] K. Kulkarni et al,"ReconNet: Non-Iterative Reconstruction of Images From Compressively Sensed Measurements", CVPR 2016. 

[2] H. Yao et al, "DR2-Net: Deep Residual Reconstruction Network for Image Compressive Sensing", arxiv, 2017



