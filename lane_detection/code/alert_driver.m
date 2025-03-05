function alert_driver(of,nf)
% This function is used to alert the driver when he is going out of the road.
% It compares the distance between the center of the image (vehicle position) and the lane lines.

of = double(of);
nf = double(nf);
[h w] = size(of);
center = double(h/2);

of_vert_prof = sum(of, 1);
nf_vert_prof = sum(nf, 1);

[of_peaks, of_peaks_locs] = findpeaks(of_vert_prof);
[nf_peaks, nf_peaks_locs] = findpeaks(nf_vert_prof);

% Take lanes position
[of_peaks, of_i] = max(of_peaks);
[nf_peaks, nf_i] = max(nf_peaks);

new_dist = abs(center-nf_peaks_locs(nf_i));
old_dist = abs(center-of_peaks_locs(of_i));

    if new_dist < old_dist && new_dist < 120
        disp("WARNING, YOU'RE GOING OUT OF ROAD")
    end
end

