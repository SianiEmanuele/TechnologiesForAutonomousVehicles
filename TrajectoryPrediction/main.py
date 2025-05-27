from truckscenes import TruckScenes
import os

cwd = os.getcwd()
dataset_path  = os.path.join(cwd, 'data/man-truckscenes' )
print(f"Dataset path: {dataset_path}")

trucksc = TruckScenes('v1.0-mini', dataset_path, True)

# def plot_trajectory(instance_token):
#     first_annotation_token = my_instance['first_annotation_token']
#     last_annotation_token = my_instance['last_annotation_token']

#     annotation_token = first_annotation_token
#     translations = []
#     ego_translations = []
#     ego_timestamps = []



#     while annotation_token != last_annotation_token:
#         annotation = trucksc.get('sample_annotation', annotation_token)
#         annotation_token = annotation['next']
#         sample_token = annotation['sample_token']
#         sample = trucksc.get('sample', sample_token)
#         closest_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])
#         ego_translation = closest_ego_pose['translation']
#         instance_translation = annotation['translation']
#         delta_translation = [instance_translation[i] - ego_translation[i] for i in range(3)]
#         translations.append(delta_translation)


#         # compensate for the movement of the ego truck
#         sample_token = annotation['sample_token']
#         sample = trucksc.get('sample', sample_token)
        
#     # plot the trajectory in x-y plane
#     import matplotlib.pyplot as plt
#     import numpy as np

#     x = [t[0] for t in translations]
#     y = [t[1] for t in translations]
#     plt.plot(x, y)
#     plt.title('Trajectory of the instance')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()


if __name__  == "__main__":
    # Get the first instance
    scene = my_scene = trucksc.scene[6]
    first_sample_token = my_scene['first_sample_token']
    print(first_sample_token)
    trucksc.render_sample(first_sample_token)


    # my_instance = trucksc.get('instance', 'a1b2c3d4e5f6g7h8i9j0')
    # plot_trajectory(my_instance['token'])