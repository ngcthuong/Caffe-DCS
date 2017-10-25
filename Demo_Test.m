clear
close all
clc
addpath('utilities')

% Dataset for testing
dataset = 'Set5';
test_images_dir = ['testsets/' dataset];
test_images = dir(test_images_dir);
test_images = test_images(3:end);
rec_method = {'ReconNet', 'DR2_Stage1', 'DR2_Stage2'};
rec_id = 2;
iter    = 100000;
blk_size = 33;

%% Select measurement rate (MR): 0.25, 0.10, 0.04 or 0.01
allMr = {'0.01', '0.04', '0.10', '0.25'};

%addpath(genpath('../../matlab'))
addpath(genpath('E:\Caffe_VS\caffe-windows-ms2\caffe-windows-ms\matlab'));
% addpath(genpath('E:\Research_DLCS\caffe-windows-bvlc\matlab'));

try
    caffe.reset_all();
catch
    caffe.reset_all();
end

gpu_id = 1; 
caffe.set_mode_gpu();
caffe.set_device(gpu_id); 

for rate_id = 4:1:numel(allMr)
    mr = allMr{rate_id}; % '0.25';
    mr_str = mr(3:end);
    
    fprintf('Measurement Rate = %s \n', mr)
    
    %% Initializations
    
    
    output_dir = ['results/' rec_method{rec_id} '/'];
    if ~exist(output_dir), mkdir(output_dir); end
    
    % Prototxt file for the selected MR    
    prototxt_file = ['solvers/' rec_method{rec_id} '/deploy_' ...
                    rec_method{rec_id}  '.prototxt'];
                
    %caffemodel = ['solvers/' rec_method{rec_id} '/subrate_0_' mr_str '/snapshot/' ...
    %              rec_method{rec_id} '_0_', mr_str, '_iter_' num2str(iter) '.caffemodel'];
    caffemodel = ['E:\Research_DLCS\caffe_deep_compressive_sensing\solvers\DR2_Stage1\subrate_0_25\snapshot\DR2_Stage1_0_25_iter_100000.caffemodel'];
    % Load the measurement matrix for the selected MR
    load(['data/CS_Meas/phi/phi_0_', mr_str, '_' num2str(blk_size^2) '.mat']);
    
    psnr = zeros(11,1);
    time_complexity = zeros(11,1);
    
    %%
    rec_folder = [ 'results/' rec_method{rec_id}  '/'  ];
    if ~exist(rec_folder), mkdir(rec_folder); end
    
    fileID = fopen([rec_folder  dataset '_rate' mr '.txt'],'w');
    PSNRCur = zeros(1,1);
    SSIMCur = zeros(1, 1);
    count_img = 0;
    
    for img_id = 1:length(test_images)
        
        %% Reading image
        image_name = test_images(img_id).name;
        input_im_nn = im2double(imread(fullfile(test_images_dir,image_name))); %Input for the ReconNet
        if(size(input_im_nn, 3) > 1)
            input_im_nn = rgb2gray(input_im_nn); 
        end
        num_blocks = ceil(size(input_im_nn,1)/blk_size)*ceil(size(input_im_nn,2)/blk_size);
        
        noMeas = size(phi, 1); 
        modify_prototxt(prototxt_file, num_blocks, noMeas);
        
        %net = caffe.Net(prototxt_file, caffemodel, 'test');
        net = caffe.Net(prototxt_file, 'test');
        net.copy_from(caffemodel); 
        
        %% Determine the size of zero pad --> it is because of weird block size of 32x32
        [row, col] = size(input_im_nn);
        row_pad = blk_size-mod(row,blk_size);
        col_pad = blk_size-mod(col,blk_size);
        
        % Do zero padding
        im_pad_nn = [input_im_nn, zeros(row,col_pad)];
        im_pad_nn = [im_pad_nn; zeros(row_pad,col+col_pad)];
        
        %% Perform block based compressive sampling
        count = 0;
        for i = 1:size(im_pad_nn,1)/blk_size
            for j = 1:size(im_pad_nn,2)/blk_size
                % Access the (i,j)th block of image
                ori_im_nn = im_pad_nn((i-1)*blk_size+1:i*blk_size,(j-1)*blk_size+1:j*blk_size);
                count = count + 1;
                %CSCNN - Take the compressed measurements of the block
                y = phi*ori_im_nn(:);
                input_deep(count,:,1,1) = y;
            end
        end
        start_time = tic;
        
        %% input_deep contains the set of CS measurements of all block,
        % net.forward compute reconstructions of all blocks parallelly
        % permutate the block
        temp = net.forward({permute(input_deep,[4 3 2 1])});
        
        %% Rearrange the reconstructions to form the final image im_comp_cscnn
        count = 0;
        for i = 1:size(im_pad_nn,1)/blk_size
            for j = 1:size(im_pad_nn,2)/blk_size
                count = count + 1;
                im_comp((i-1)*blk_size+1:i*blk_size,(j-1)*blk_size+1:j*blk_size) = temp{1}(:,:,1,count);
            end
        end
        time_complexity(img_id) = toc(start_time);
        
        % croping image
        rec_im = im_comp(1:row, 1:col,:);
        [~,name,~] = fileparts(image_name);
        
        %% Evaluation
        rec_folder2 = [ 'results/'  rec_method{rec_id} '/' dataset '_img_0_' mr_str ];
        
        if ~exist(rec_folder2), mkdir(rec_folder2); end
        imwrite(rec_im, [rec_folder2 '/' name '_rate' mr '.png']);
        
        
        count_img = count_img + 1;
        [PSNRCur(count_img), SSIMCur(count_img)] = Cal_PSNRSSIM(im2uint8(input_im_nn),im2uint8(rec_im),0,0);
        
        clear im_comp temp input_deep
        
        fprintf('\n %15s: PSNR = %6.2f dB, SSIM = %6.3f , Time = %f  seconds\n', image_name, ...
            PSNRCur(count_img), SSIMCur(count_img), time_complexity(count_img))    ;
        
        fprintf(fileID, '%20s: PSNR = %6.2f dB, SSIM = %6.3f , Time = %f  seconds\n', image_name, ...
            PSNRCur(count_img), SSIMCur(count_img), time_complexity(count_img));
        
    end
    fprintf(fileID, '%20s: PSNR = %6.2f dB, SSIM = %6.3f , Time = %f  seconds\n', 'Average', ...
        mean(PSNRCur), mean(SSIMCur), mean(time_complexity));
    fclose(fileID);
end;
fprintf(['\n Finished Testing ', dataset '\n'])
