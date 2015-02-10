function visualiseCovars()

encoder = encoderFactory();

weights = encoder.get_weights();

w = weights(8).weights{1};
cov = w'*w;

imdb = load('imdb-ILSVRC2012.mat');
imdb = imdb.imdb;

fig_h = figure('WindowButtonMotionFcn', @hoverCallback);
axes_h = axes;
imagesc(cov, 'Parent', axes_h);

title('ImageNet w-vects Covariance');

function hoverCallback(src, evt)
    % Grab the x & y axes coordinate where the mouse is
    mousePoint = get(axes_h, 'CurrentPoint');
    mouse_x = floor(mousePoint(1,1));
    mouse_y = floor(mousePoint(1,2));

    N_c = length(imdb.classes.description);
    mouse_x = max([1, mouse_x]);
    mouse_x = min([N_c, mouse_x]);

    mouse_y = max([1, mouse_y]);
    mouse_y = min([N_c, mouse_y]);

    xlabel(axes_h, imdb.classes.description{mouse_x});
    ylabel(axes_h, imdb.classes.description{mouse_y});
end
end
