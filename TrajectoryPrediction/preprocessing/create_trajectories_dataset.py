from scipy.spatial.transform import Rotation as R
from truckscenes import TruckScenes
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract_trajectory(annotations: list, plot=True):
    """
    Extracts the trajectory of an instance in the ego vehicle's frame of reference.
    
    Parameters:
    - annotations: list of annotations for the instance.
    - plot: boolean indicating whether to plot the trajectory or not.

    Returns:
    - x_rel: list of x coordinates of the trajectory in the ego vehicle's frame.
    - y_rel: list of y coordinates of the trajectory in the ego vehicle's frame.
    """
    relative_positions = []
    ego_positions = []

    for annotation in annotations:
        sample_token = annotation['sample_token']
        sample = trucksc.get('sample', sample_token)

        # Ego position and rotation
        ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])
        ego_translation = ego_pose['translation']
        ego_rotation = ego_pose['rotation']  # quaternion [w, x, y, z]

        # Quaternion representation for scipy
        q = [ego_rotation[1], ego_rotation[2], ego_rotation[3], ego_rotation[0]]
        ego_rot_inv = R.from_quat(q).inv()

        # Coordinate dell'oggetto (istanza) nel mondo
        instance_translation = annotation['translation']

        # Traslazione relativa (coordinate mondo)
        rel_translation = [instance_translation[i] - ego_translation[i] for i in range(3)]

        # Rotazione inversa per portare nel frame dell’ego vehicle
        rel_in_ego = ego_rot_inv.apply(rel_translation)

        # Salva solo x e y (piano orizzontale)
        relative_positions.append((rel_in_ego[0], rel_in_ego[1]))
        ego_positions.append((0.0, 0.0))  # ego è sempre all'origine nel suo sistema



    # Estrai coordinate per il grafico
    x_rel, y_rel = zip(*relative_positions)
    x_ego, y_ego = zip(*ego_positions)

    # Plot
    if plot:
        plt.plot(x_rel, y_rel, label="Trajectory in ego's frame", color='blue')
        plt.scatter(x_rel[0], y_rel[0], color='green', label='Inizio')
        plt.scatter(x_rel[-1], y_rel[-1], color='orange', label='Fine')
        # ego position wiFth a red cross
        plt.scatter(x_ego[0], y_ego[0], color='red', marker='x', label='Posizione Ego')
        plt.title('Instance Trajectory in Ego Frame')
        plt.xlabel('x (m) - frame ego')
        plt.ylabel('y (m) - frame ego')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return x_rel, y_rel

def get_all_scenes_tokens(trucksc):
    """
    Extracts all scenes from the TruckScenes dataset.
    """
    scenes = []
    for scene in trucksc.scene:
        scenes.append(scene)
    return scenes

def get_all_samples(trucksc, scenes: list):
    """
    Extracts all samples from the given scenes.

    Parameters:
    - scenes: list of scenes to extract samples from.

    Returns:
    - samples: list of samples extracted from the scenes.
    """
    samples = []
    for scene in scenes:
        completed = False
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        sample = trucksc.get('sample', first_sample_token)
        samples.append(sample)

        while not completed:
            sample = trucksc.get('sample', sample_token)
            sample_token = sample['next']
            if sample_token == scene['last_sample_token']:
                completed = True
            samples.append(sample)
    return samples

def get_sample_annotations(trucksc, samples):
    """
    Extracts all annotations from the given samples.
    """
    annotations = []
    for sample in samples:
        for annotation_token in sample['anns']:
            annotation = trucksc.get('sample_annotation', annotation_token)
            annotations.append(annotation)
    return annotations

def get_selected_vehicles(trucksc):
    """
    Extracts all humans and vehicles from the TruckScenes dataset.
    
    Parameters:
    - trucksc: TruckScenes object.

    Returns:
    - all_instances: list of all instances (humans and vehicles).
    """
    accepted_categories = ['vehicle.car', 'vehicle.truck', 'vehicle.motorcycle', 'vehicle.bus']
    vehicles = []
    for instance in trucksc.instance:
        category_token = instance['category_token']
        category = trucksc.get('category', category_token)
        if category['name'] in accepted_categories:
            vehicles.append(instance)
    
    # combine humans and vehicles

    return vehicles

def get_all_annotations(trucksc, instances: list):
    """
    Extracts all annotations for each instance in the TruckScenes dataset.
    
    Parameters:
    - trucksc: TruckScenes object.
    - instances: list of instances to extract annotations for.

    Returns:
    - annotations: dictionary with instance tokens as keys and lists of annotations as values.
    """
    annotations = dict()
    for instance in instances:
        instance_token = instance['token']
        first_annotation_token = instance['first_annotation_token']
        last_annotation_token = instance['last_annotation_token']

        annotation_token = first_annotation_token
        annotations[instance_token] = []

        while True:
            annotation = trucksc.get('sample_annotation', annotation_token)
            annotations[instance_token].append(annotation)

            if annotation_token == last_annotation_token:
                annotations[instance_token].append(annotation)
                break
            annotation_token = annotation['next']

    return annotations

def create_dataframe(instances_annotations: dict, output_dir: str):
    dataset = pd.DataFrame(columns=['instance_token', 'category_name', 'x_rel', 'y_rel', 'rot_q0', 'rot_q1', 'rot_q2', 'rot_q3' , 'timestamp'])
    num_instances = len(instances_annotations)
    for i, (instance_token, annotations) in enumerate(instances_annotations.items()):
        # log every 100 instances
        if i % 100 == 0 or i == 0:
            print(f"Processing instance {i}/{num_instances} ({instance_token})")
        # Get the category name
        instance = trucksc.get('instance', instance_token)
        category_token = instance['category_token']
        category = trucksc.get('category', category_token)
        category_name = category['name']

        trajectories = extract_trajectory(annotations, plot=False)
        x_rel, y_rel = trajectories
        v_x = []
        v_y = []
        
        timestamps = []
        rot_q0 = []
        rot_q1 = []
        rot_q2 = []
        rot_q3 = []
        
        
        for annotation in annotations:
            sample_token = annotation['sample_token']
            sample = trucksc.get('sample', sample_token)
            timestamps.append(sample['timestamp'])
            rot_q0.append(annotation['rotation'][0])
            rot_q1.append(annotation['rotation'][1])
            rot_q2.append(annotation['rotation'][2])
            rot_q3.append(annotation['rotation'][3])

        # Calculate the velocity
        v_x.append(0)
        v_y.append(0)
        for j in range(len(x_rel) - 1):
            delta_t = timestamps[j + 1] - timestamps[j]
            if delta_t > 0:
                v_x.append((x_rel[j + 1] - x_rel[j]) / delta_t)
                v_y.append((y_rel[j + 1] - y_rel[j]) / delta_t)
            else:
                v_x.append(0)
                v_y.append(0)
        
            
        # Create a dataframe for the instance
        instance_df = pd.DataFrame({
            'instance_token': [instance_token] * len(x_rel),
            'category_name': [category_name] * len(x_rel),
            'x_rel': x_rel,
            'y_rel': y_rel,
            'v_x': v_x,
            'v_y': v_y,
            'rot_q0': rot_q0,
            'rot_q1': rot_q1,
            'rot_q2': rot_q2,
            'rot_q3': rot_q3,
            'timestamp': timestamps
        })

        # Append the instance dataframe to the main dataframe
        dataset = pd.concat([dataset, instance_df], ignore_index=True)

    # Map category names to integers
    category_mapping = {name: idx for idx, name in enumerate(dataset['category_name'].unique())}
    dataset['category_id'] = dataset['category_name'].map(category_mapping)

    # one hot encode the category_id
    dataset = pd.get_dummies(dataset, columns=['category_id'], prefix='category')

    # remove the category_name column
    dataset.drop(columns=['category_name'], inplace=True)


    # Stampa il dizionario di mapping
    print("\nDizionario di mapping:")
    for k, v in category_mapping.items():
        print(f"{v}: {k}")

    # Save the dataset to a csv file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset.to_csv(os.path.join(output_dir, 'trajectories_dataset.csv'), index=False)
    print(f"Dataset saved to {os.path.join(output_dir, 'trajectories_dataset.csv')}")


if __name__ == "__main__":
    cwd = os.getcwd()
    dataset_path  = os.path.join(cwd, 'TrajectoryPrediction/data/man-truckscenes' )
    version = 'v1.0-trainval'
    trucksc = TruckScenes(version, dataset_path, True)

    # Extract all different instances
    instances = get_selected_vehicles(trucksc)
    print(f"Extracted {len(instances)} instances.")

    # Extract all annotations for each instance
    instances_annotations = get_all_annotations(trucksc, instances)
    
    # remove instances with less than 20 annotations
    instances_annotations = {k: v for k, v in instances_annotations.items() if len(v) > 20}
    print(f"Extracted {len(instances_annotations)} instances with more than 20 annotations.")

    save_path = os.path.join(cwd, 'TrajectoryPrediction', 'processed_data', version)
    # create dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    create_dataframe(instances_annotations, save_path)

    # plot some trajectories
    for i, (instance_token, annotations) in enumerate(instances_annotations.items()):
        if i < 5:  # Plot only the first 5 instances
            print(f"Plotting trajectory for instance {instance_token}")
            extract_trajectory(annotations, plot=True)
        else:
            break

    
    
            