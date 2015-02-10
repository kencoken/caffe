function code = encode(obj, im)

    assert(obj.net_handle_ > 0);
    assert(isa(im, 'single'));
    im = im*255.0;

    % get input images
    fixed_prep_image = (obj.caffe_version >= 1.1);

    input_data = ...
        {obj.augmentation_helper_.prepareImage(im, ...
                                               'mean_img', obj.average_image, ...
                                               'preproc_dup_grey', obj.preproc_dup_grey, ...
                                               'fixed_prep_image', fixed_prep_image)};

    % pass through net
    %fprintf('Forwarding with handle: %d\n', obj.net_handle_);
    code = caffe('forward', input_data, obj.output_blob_names, obj.net_handle_);

    code = cellfun(@(x) squeeze(x), code, 'UniformOutput', false);
    
%     net_code_dims = obj.get_net_output_dim_();
%     for i = 1:length(code)
%         code{i} = reshape(code{i}, [net_code_dims(i), numel(code{i})/net_code_dims(i)]);
%     end

    % collate and normalise result
    code = cellfun(@(x) obj.augmentation_helper_.transformCodes(x), code, 'UniformOutput', false);
    
    if length(code) == 1
        code = code{1};
    end

end
