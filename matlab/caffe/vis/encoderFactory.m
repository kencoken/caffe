function encoder = encoderFactory()
%ENCODERFACTORY Summary of this function goes here
%   Detailed explanation goes here

addpath('../');

model_dir = '/data/ken/caffe_models/CNN_M_128';
param_file = sprintf('%s/VGG_CNN_M_128_deploy.prototxt', model_dir);
model_file = sprintf('%s/VGG_CNN_M_128.caffemodel', model_dir);
average_image = sprintf('%s/VGG_mean.mat', model_dir);
encoder = featpipem.directencode.ConvNetEncoder(param_file, model_file, ...
                                                average_image, ...
                                                'output_blob_names', {'fc7','fc8'});

encoder.augmentation = 'aspect_corners';
encoder.augmentation_collate = 'sum';

encoder.set_backend('gpu', 0);

end
