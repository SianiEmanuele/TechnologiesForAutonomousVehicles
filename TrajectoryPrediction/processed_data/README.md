# MAN-TruckScenes Trajectory Preprocessing

This folder contains a preprocessed version of the [MAN-TruckScenes](https://truckscenes.in.tum.de/) dataset for trajectory prediction tasks. The preprocessing pipeline extracts 2D trajectories of human and vehicle instances relative to the ego vehicle's frame of reference and exports them to a CSV file.

## Overview

The script performs the following steps:
1. Loads the MAN-TruckScenes dataset using the `TruckScenes` API.
2. Filters all instances that are either vehicles or humans.
3. For each instance:
   - Retrieves all associated annotations.
   - Transforms each annotation's world position into the ego vehicle's coordinate frame.
   - Extracts additional information such as timestamp and 2D rotation.
4. Filters out short trajectories (less than 20 annotations).
5. Saves the processed data in a CSV format suitable for training and evaluation.

## Output Format

The resulting CSV file (`trajectories_dataset.csv`) contains the following columns:

| Column Name     | Description                                        |
|------------------|----------------------------------------------------|
| instance_token   | Unique ID of the instance                          |
| x_rel            | X position in ego vehicle's frame (in meters)      |
| y_rel            | Y position in ego vehicle's frame (in meters)      |
| rot_x            | Rotation (quaternion x component)                  |
| rot_y            | Rotation (quaternion y component)                  |
| timestamp        | Timestamp of the annotation                        |
| category_id      | Integer representing the instance category         |

## Category ID Mapping

Below is the mapping used to convert category names to integer labels:

0.  human.pedestrian.adult  
1.  human.pedestrian.child  
2.  human.pedestrian.stroller  
3.  human.pedestrian.construction_worker  
4.  vehicle.car  
5.  vehicle.truck  
6.  vehicle.trailer  
7.  vehicle.ego_trailer  
8.  vehicle.motorcycle  
9.  vehicle.bicycle  
10. vehicle.other  
11. vehicle.bus.rigid  
12. vehicle.construction  
13. vehicle.train  

