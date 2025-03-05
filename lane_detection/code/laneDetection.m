clc;
clear all;
close all;
%% camera sensor parameters
camera = struct('ImageSize',[480 640],'PrincipalPoint',[320 240],...
                'FocalLength',[320 320],'Position',[1.8750 0 1.2000],...
                'PositionSim3d',[0.5700 0 1.2000],'Rotation',[0 0 0],...
                'LaneDetectionRanges',[6 30],'DetectionRanges',[6 50],...
                'MeasurementNoise',diag([6,1,1]));
focalLength    = camera.FocalLength;
principalPoint = camera.PrincipalPoint;
imageSize      = camera.ImageSize;
% mounting height in meters from the ground
height         = camera.Position(3);  
% pitch of the camera in degrees
pitch          = camera.Rotation(2);  
            
camIntrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);
sensor        = monoCamera(camIntrinsics, height, 'Pitch', pitch);

%% define area to transform
distAheadOfSensor = 30; % in meters, as previously specified in monoCamera height input
spaceToOneSide    = 8;  % all other distance quantities are also in meters
bottomOffset      = 6;
outView   = [bottomOffset, distAheadOfSensor, -spaceToOneSide, spaceToOneSide]; % [xmin, xmax, ymin, ymax]
outImageSize = [NaN, 250]; % output image width in pixels; height is chosen automatically to preserve units per pixel ratio

birdsEyeConfig = birdsEyeView(sensor, outView, outImageSize);

videoReader = VideoReader('input/driftLeft.mp4');


%% process video frame by frame
i= 0;
figure
while hasFrame(videoReader)
    frame = readFrame(videoReader); % get the next video frame
    
    birdsEyeImage = transformImage(birdsEyeConfig, frame);
    birdsEyeImage = rgb2gray(birdsEyeImage);
    figure(1)
    imshow(birdsEyeImage)
    [h,w] = size(birdsEyeImage);
    
    crop_width = round((3/4) * w);
    x_start = round((w - crop_width) / 2); % Offset da sinistra
    
    % Crops image
    birdsEyeImage = imcrop(birdsEyeImage, [x_start, 0, crop_width - 1, h - 1]);
    

    if double(max(birdsEyeImage, [], 'all')) == 0
        continue
    else
        if i==0
            th = double((max(birdsEyeImage, [], 'all') +min(birdsEyeImage, [], 'all'))) / double(2);
        end
        th = calculate_threshold(th, birdsEyeImage);
        bin_img = binarize_img(th, birdsEyeImage);
        figure(2)
        imshow(bin_img);
        figure(3);
        plot_hist(bin_img);
        
        if i > 3
            alert_driver(old_img, bin_img)
        end
        old_img = bin_img;
        
        i = i +1;
    end
end