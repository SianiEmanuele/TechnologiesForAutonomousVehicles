function [img] = binarize_img(th,img)
% Function to binarize image using a threshold
[h,w] = size(img);
for i = 1:h
    for j = 1:w
        pix = img(i, j);
        if pix > th
            img(i,j) = 255;
        else
            img(i,j) = 0;
        end
    end
end

