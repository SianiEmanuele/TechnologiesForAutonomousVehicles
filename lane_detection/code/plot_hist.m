function plot_hist(img)
%PLOT_HIST Summary of this function goes here
%   Detailed explanation goes here
    clf
    vertical_profile = sum(img, 1); 
    hold on
    plot(vertical_profile, 'r', 'LineWidth', 1.5)
    hold off
    xlabel('Pixel position');
    ylabel('Columns sum');
    title("White pixels' distribution");
    grid on;
    drawnow;
end

