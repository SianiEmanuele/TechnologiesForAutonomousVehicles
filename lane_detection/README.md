# Lane Detection System

![Demo GIF](demo.gif)  

## Overview
This project implements a toy **Lane Assistant** using **MATLAB**, designed to analyze preprocess video images taken from onboard cameras, detect lanes and alert the driver if it is getting close to the lane.

It exploits **Computer Vision Toolbox** for transforming the image to birds' eye view. Then the image is binarized exploiting an iteratively calculated threshold, making the lane detection easy and computationally cheap.

## Prerequisites
To run this project, ensure you have the following **MATLAB toolboxes** installed:

- **Computer Vision Toolbox**
- **Signal Processing Toolbox** (Required for `findpeaks` function)
