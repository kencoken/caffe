function visuliseCovars()

encoder = encoderFactory();
                                            
weights = encoder.get_weights();

w = weights(8).weights{1};
cov = w'*w;

fig_h = figure('WindowButtonMotionFcn', @hoverCallback);
axes_h = axes;
imagesc(cov, 'Parent', axes_h);

imdb = load('imdb-ILSVRC2012.mat');
imdb = imdb.imdb;

function hoverCallback(src, evt)
    % Grab the x & y axes coordinate where the mouse is
    mousePoint = get(axes_h, 'CurrentPoint');
    mouse_x = floor(mousePoint(1,1));
    %mouse_y = mousePoint(1,2);
    
    N_c = length(imdb.classes.description);
    if mouse_x >= 1 && mouse_x <= N_c
        title(imdb.classes.description{mouse_x});
    end
end

end