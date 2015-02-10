function feats = compTrainFeats()

encoder = encoderFactory();

class_dirs = dir('postrainimgs');
class_dirs = class_dirs(3:end);

classes = arrayfun(@(x) regexp(x.name,'{([a-zA-Z ]+)}','tokens'), class_dirs, 'UniformOutput', false);
classes = cellfun(@(x) x{1}, classes, 'UniformOutput', false);
classes = cellfun(@(x) x{1}, classes, 'UniformOutput', false);

feats = repmat(struct(), length(classes), 1);

for i = 1:length(classes)
    fprintf('Processing class %d of %d...\n', i, length(classes));
    class_images = dir(fullfile('postrainimgs', class_dirs(i).name, '*.jpg'));

    output_dims = encoder.get_output_dim();
    output_blob_names = encoder.output_blob_names;

    if length(output_blob_names) > 1
        assert(iscell(output_dims));

        codes = struct();
        for blb_idx = 1:length(output_dims)
            codes.(output_blob_names{blb_idx}) = ones(output_dims{blb_idx}, length(class_images), 'single');
        end
    else
        codes = ones(output_dims, length(class_images), 'single');
    end

    for im_idx = 1:length(class_images)
        imfile = fullfile('postrainimgs', class_dirs(i).name, class_images(im_idx).name);
        fprintf('Processing %s...', imfile);

        im = imread(imfile);
        im = featpipem.utility.standardizeImage(im);

        a_tic = tic;
        code = encoder.encode(im);
        if length(output_blob_names) > 1
            for blb_idx = 1:length(code)
                codes.(output_blob_names{blb_idx})(:, im_idx) = code{blb_idx};
            end
        else
            codes(:, im_idx) = code;
        end
        fprintf('Computed in %d seconds\n', toc(a_tic));
    end

    feats(i).name = classes{i};
    feats(i).codes = codes;
    feats(i).paths = arrayfun(@(x) x.name, class_images, 'UniformOutput', false);
end

end
