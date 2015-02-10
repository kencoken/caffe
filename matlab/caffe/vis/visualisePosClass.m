function visualisePosClass(class_name, sorted)

if nargin < 2 || isempty(sorted)
    sorted = true;
end

cls_feats = load('posfeats.mat');
cls_feats = cls_feats.feats;

cls_idx = -1;
for ci = 1:length(cls_feats)
    if strcmp(cls_feats(ci).name, class_name)
        cls_idx = ci;
        break;
    end
end
if cls_idx == -1, error('Class could not be found'); end

the_cls_feats = cls_feats(cls_idx).codes.fc8;
clear cls_feats;

imdb = load('imdb-ILSVRC2012.mat');
imdb = imdb.imdb;
class_descs = imdb.classes.description;
clear imdb;

% reorder imagenet classes per mean score for that class if sorted is true
if sorted
    mean_feats = sum(the_cls_feats, 2)./size(the_cls_feats,2);

    [~, sort_idxs] = sort(mean_feats, 'descend');

    the_cls_feats = the_cls_feats(sort_idxs,:);
    class_descs = class_descs(sort_idxs);
    clear mean_feats;
end

fig_h = figure('WindowButtonMotionFcn', @hoverCallback);
axes_h = axes;
imagesc(the_cls_feats, 'Parent', axes_h);

title(sprintf('Class scores: %s', class_name));

function hoverCallback(src, evt)
    % Grab the x & y axes coordinate where the mouse is
    mousePoint = get(axes_h, 'CurrentPoint');
    mouse_y = floor(mousePoint(1,2));

    N_c = length(class_descs);
    mouse_y = max([1, mouse_y]);
    mouse_y = min([N_c, mouse_y]);

    ylabel(axes_h, class_descs{mouse_y});
end

end
