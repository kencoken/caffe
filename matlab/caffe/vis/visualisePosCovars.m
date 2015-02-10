function visualisePosCovars(sorted)

if nargin < 1 || isempty(sorted)
    sorted = false;
end

encoder = encoderFactory();

cls_feats = load('posfeats.mat');
cls_feats = cls_feats.feats;

mean_feats = zeros(size(cls_feats(1).codes.fc7,1), length(cls_feats), 'single');
for ci = 1:length(cls_feats)
    mean_feats(:,ci) = sum(cls_feats(ci).codes.fc7, 2)./size(cls_feats(ci).codes.fc7,2);
end

weights = encoder.get_weights();

w = weights(8).weights{1};
cov = mean_feats'*w;

% reorder positive image classes in order of max activation across imagenet
% classes
[~, max_imagenet_idxs] = max(cov,[],2);
[~, max_idxs] = sort(max_imagenet_idxs, 'ascend');
cls_feats = cls_feats(max_idxs);
mean_feats = mean_feats(:,max_idxs);
cov = cov(max_idxs,:);

% reorder imagenet classes per row if sorted is true
if sorted
    sort_idxs = zeros(size(cov,1), size(cov,2));
    for ci = 1:size(cov,1)
        [cov(ci,:), sort_idxs(ci,:)] = sort(cov(ci,:), 'descend');
    end
end

imdb = load('imdb-ILSVRC2012.mat');
imdb = imdb.imdb;

fig_h = figure('WindowButtonMotionFcn', @hoverCallback);
axes_h = axes;
imagesc(cov, 'Parent', axes_h);

title(sprintf('Mean class w-vects vs. ImageNet w-vects (sorted: %d)', sorted));

function hoverCallback(src, evt)
    % Grab the x & y axes coordinate where the mouse is
    mousePoint = get(axes_h, 'CurrentPoint');
    mouse_x = floor(mousePoint(1,1));
    mouse_y = floor(mousePoint(1,2));

    N_c = length(imdb.classes.description);
    mouse_x = max([1, mouse_x]);
    mouse_x = min([N_c, mouse_x]);

    NT_c = length(cls_feats);
    mouse_y = max([1, mouse_y]);
    mouse_y = min([NT_c, mouse_y]);

    if ~sorted
        xlabel(axes_h, imdb.classes.description{mouse_x});
    else
        xlabel(axes_h, imdb.classes.description{sort_idxs(mouse_y, mouse_x)});
    end

    ylabel(axes_h, cls_feats(mouse_y).name);
end

end
