function codes = compILSVRCFeats(set_name)

assert(nargin >= 1);

encoder = encoderFactory();

% start parsing set_name

switch upper(set_name)
    case 'TRAIN'
        set_idx = 1;
    case 'VAL'
        set_idx = 2;
    case 'TEST'
        set_idx = 3;
    otherwise
        error('Could not parse set_name');
end

disp 'Loading ILSVRC IMDB...';
imdb = load('imdb-ILSVRC2012.mat');
imdb = imdb.imdb;

disp 'Extracting IDs and names...';
idxs = (imdb.images.set == set_idx);
ids = imdb.images.id(idxs);
names = imdb.images.name(idxs);

output_dims = encoder.get_output_dim();
output_blob_names = encoder.output_blob_names;

if length(output_blob_names) > 1
    assert(iscell(output_dims));

    codes = struct();
    for blb_idx = 1:length(output_dims)
        codes.(output_blob_names{blb_idx}) = ones(output_dims{blb_idx}, length(names), 'single');
    end
else
    codes = ones(output_dims, length(names), 'single');
end

for im_idx = 1:length(names)
    fprintf('Computing feature %d/%d: %s...\n', im_idx, length(names), names{im_idx});

    im = imread(fullfile(imdb.imageDir, names{im_idx}));
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

save(sprintf('ILSVRC_feats_%s.mat', lower(set_name)), 'codes', '-v7.3');

end
