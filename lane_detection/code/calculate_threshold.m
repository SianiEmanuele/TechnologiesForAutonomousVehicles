function [new_th] = calculate_threshold(old_th,img)
% Iteratively calculates the threshold for binarization

[h,w] = size(img);
img = double(img);
n_pix_above = 0;
n_pix_below = 0;
sum_pix_above = 0;
sum_pix_below = 0;
new_th = 10000;

while not (abs(new_th - old_th) < 0.01)
    old_th = new_th;
    for i = 1:h
        for j = 1:w
            pix = img(i, j);
            if pix > old_th
                n_pix_above = n_pix_above + 1;
                sum_pix_above = sum_pix_above + pix;
            else
                n_pix_below = n_pix_below + 1;
                sum_pix_below = sum_pix_below + pix;
            end
        end
    end
    % Evita divisioni per zero
    if n_pix_above > 0
        ga = sum_pix_above / n_pix_above;
    else
        ga = 0;
    end
    
    if n_pix_below > 0
        gb = sum_pix_below / n_pix_below;
    else
        gb = 0;
    end
        new_th = (ga + gb) / 2;
end
end