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
        # ego position with a red cross
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

def get_all_humans_and_vehicles(trucksc):
    """
    Extracts all humans and vehicles from the TruckScenes dataset.
    
    Parameters:
    - trucksc: TruckScenes object.

    Returns:
    - all_instances: list of all instances (humans and vehicles).
    """
    humans = []
    vehicles = []
    for instance in trucksc.instance:
        category_token = instance['category_token']
        category = trucksc.get('category', category_token)
        if 'vehicle' in category['name']:
            vehicles.append(instance)
        elif 'human' in category['name']:
            humans.append(instance)
    
    # combine humans and vehicles
    all_instances = humans + vehicles
    return all_instances

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
    dataset = pd.DataFrame(columns=['instance_token', 'category_name', 'x_rel', 'y_rel', 'rot_x', 'rot_y',  'timestamp'])

    for instance_token, annotations in instances_annotations.items():
        # Get the category name
        instance = trucksc.get('instance', instance_token)
        category_token = instance['category_token']
        category = trucksc.get('category', category_token)
        category_name = category['name']

        trajectories = extract_trajectory(annotations, plot=False)
        x_rel, y_rel = trajectories

        timestamps = []
        rot_x = []
        rot_y = []
        for annotation in annotations:
            sample_token = annotation['sample_token']
            sample = trucksc.get('sample', sample_token)
            timestamps.append(sample['timestamp'])
            rot_x.append(annotation['rotation'][0])
            rot_y.append(annotation['rotation'][1])
            

        # Create a dataframe for the instance
        instance_df = pd.DataFrame({
            'instance_token': [instance_token] * len(x_rel),
            'category_name': [category_name] * len(x_rel),
            'x_rel': x_rel,
            'y_rel': y_rel,
            'timestamp': timestamps,
            'rot_x': rot_x,
            'rot_y': rot_y
        })

        # Append the instance dataframe to the main dataframe
        dataset = pd.concat([dataset, instance_df], ignore_index=True)

    # Map category names to integers
    category_mapping = {name: idx for idx, name in enumerate(dataset['category_name'].unique())}
    dataset['category_id'] = dataset['category_name'].map(category_mapping)

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
    print(f"Dataset path: {dataset_path}")

    trucksc = TruckScenes('v1.0-mini', dataset_path, True)

    # scenes = get_all_scenes_tokens(trucksc)
    # print(f"Extracted {len(scenes)} scenes.")

    # samples = get_all_samples(trucksc, scenes)
    # print(f"Extracted {len(samples)} samples.")

    # # Extract all annotations
    # annotations = get_sample_annotations(trucksc, samples)
    # print(f"Extracted {len(annotations)} annotations.")

    # Extract all different instances
    instances = get_all_humans_and_vehicles(trucksc)
    print(f"Extracted {len(instances)} instances.")

    # Extract all annotations for each instance
    instances_annotations = get_all_annotations(trucksc, instances)
    
    # remove instances with less than 20 annotations
    instances_annotations = {k: v for k, v in instances_annotations.items() if len(v) > 20}
    print(f"Extracted {len(instances_annotations)} instances with more than 20 annotations.")

    save_path = os.path.join(cwd, 'TrajectoryPrediction', 'processed_data')
    create_dataframe(instances_annotations, save_path)

    
    
            