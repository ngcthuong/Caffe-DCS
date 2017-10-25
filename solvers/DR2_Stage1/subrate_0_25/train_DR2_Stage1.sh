rm DR2_Stage1.log
/mnt/e/Research_DLCS/caffe-windows-bvlc/build/tools/Release/caffe.exe train --solver=solver_DR2_Stage1.prototxt --gpu 1|  tee -a DR2_Stage1.log