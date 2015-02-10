function compILSVRCFeats(set_name)

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

codes = ones(encoder.get_output_dim(), length(names), 'single');
for i = 1:length(names)
    fprintf('Computing feature %d/%d: %s...\n', i, length(names), names{i});

    im = imread(fullfile(imdb.imageDir, names{i}));
    im = featpipem.utility.standardizeImage(im);

    a_tic = tic;
    codes(:, im_idx) = encoder.encode(im);
    fprintf('Computed in %d seconds\n', toc(a_tic));
end

save(sprintf('ILSVRC_feats_%s.mat', lower(set_name)), 'codes');

end
