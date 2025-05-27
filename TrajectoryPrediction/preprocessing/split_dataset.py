import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



if __name__ == "__main__":
    cwd = os.getcwd()
    version = 'v1.0-mini'
    raw_dataset_path = os.path.join(cwd,f'TrajectoryPrediction', 'processed_data', version, 'trajectories_dataset.csv')

    # Load the dataset
    dataset = pd.read_csv(raw_dataset_path)

    print(f'\nORIGINAL DATASET')
    print(f'Dataset shape: {dataset.shape}')

    print(dataset.describe())

    # get the unique instance tokens
    instance_tokens = dataset['instance_token'].unique()
    # divide into train, val and test sets
    train_size = int(0.7 * len(instance_tokens))
    val_size = int(0.15 * len(instance_tokens))
    test_size = len(instance_tokens) - train_size - val_size
    print(f'\nTRAINING, VALIDATION AND TEST SIZES')
    print(f'Training instances: {train_size}')
    print(f'Validation instances: {val_size}')
    print(f'Test instances: {test_size}')

    # Shuffle the instance tokens and split them into train, val and test sets
    np.random.seed(77) if 'trainval' in version else np.random.seed(33)
    np.random.shuffle(instance_tokens)
    train_instance_tokens = instance_tokens[:train_size]
    val_instance_tokens = instance_tokens[train_size:train_size + val_size]
    test_instance_tokens = instance_tokens[train_size + val_size:]

    print(f'Training set shape: {len(train_instance_tokens)}')
    print(f'Validation set shape: {len(val_instance_tokens)}')
    print(f'Test set shape: {len(test_instance_tokens)}')
    # Save the datasets into train, val and test folders
    train_dir = os.path.join(cwd, 'TrajectoryPrediction', 'processed_data', version, 'train')
    val_dir = os.path.join(cwd, 'TrajectoryPrediction', 'processed_data', version, 'val')
    test_dir = os.path.join(cwd, 'TrajectoryPrediction', 'processed_data', version, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    # save the train, val and test sets
    train_df = dataset[dataset['instance_token'].isin(train_instance_tokens)]
    val_df = dataset[dataset['instance_token'].isin(val_instance_tokens)]
    test_df = dataset[dataset['instance_token'].isin(test_instance_tokens)]
    
    # normalize train data
    train_x_mean, train_x_std = train_df['x_rel'].mean(), train_df['x_rel'].std()
    train_y_mean, train_y_std = train_df['y_rel'].mean(), train_df['y_rel'].std()
    # train_rot_q0_mean, train_rot_q0_std = train_df['rot_q0'].mean(), train_df['rot_q0'].std()
    # train_rot_q1_mean, train_rot_q1_std = train_df['rot_q1'].mean(), train_df['rot_q1'].std()
    # train_rot_q2_mean, train_rot_q2_std = train_df['rot_q2'].mean(), train_df['rot_q2'].std()
    # train_rot_q3_mean, train_rot_q3_std = train_df['rot_q3'].mean(), train_df['rot_q3'].std()
    # train_v_x_mean, train_v_x_std = train_df['v_x'].mean(), train_df['v_y'].std()
    # train_v_y_mean, train_v_y_std = train_df['v_x'].mean(), train_df['v_y'].std()
    train_speed_mean, train_speed_std = train_df['speed'].mean(), train_df['speed'].std()
    train_heading_mean, train_heading_std = train_df['heading'].mean(), train_df['heading'].std()

    train_df['x_rel'] = (train_df['x_rel'] - train_x_mean) / train_x_std
    train_df['y_rel'] = (train_df['y_rel'] - train_y_mean) / train_y_std
    # train_df['rot_q0'] = (train_df['rot_q0'] - train_rot_q0_mean) / train_rot_q0_std
    # train_df['rot_q1'] = (train_df['rot_q1'] - train_rot_q1_mean) / train_rot_q1_std
    # train_df['rot_q2'] = (train_df['rot_q2'] - train_rot_q2_mean) / train_rot_q2_std
    # train_df['rot_q3'] = (train_df['rot_q3'] - train_rot_q3_mean) / train_rot_q3_std
    # train_df['v_x'] = (train_df['v_x'] - train_v_x_mean) / train_v_x_std
    # train_df['v_y'] = (train_df['v_y'] - train_v_y_mean) / train_v_y_std
    train_df['speed'] = (train_df['speed'] - train_speed_mean) / train_speed_std
    train_df['heading'] = (train_df['heading'] - train_heading_mean) / train_heading_std


    # describe the train data
    print(f'\nNORMALIZED TRAIN DATA')
    print(train_df.describe())

    # normalize val data
    val_x_mean, val_x_std = val_df['x_rel'].mean(), val_df['x_rel'].std()
    val_y_mean, val_y_std = val_df['y_rel'].mean(), val_df['y_rel'].std()
    # val_rot_q0_mean, val_rot_q0_std = val_df['rot_q0'].mean(), val_df['rot_q0'].std()
    # val_rot_q1_mean, val_rot_q1_std = val_df['rot_q1'].mean(), val_df['rot_q1'].std()
    # val_rot_q2_mean, val_rot_q2_std = val_df['rot_q2'].mean(), val_df['rot_q2'].std()
    # val_rot_q3_mean, val_rot_q3_std = val_df['rot_q3'].mean(), val_df['rot_q3'].std()
    # val_v_x_mean, val_v_x_std = val_df['v_x'].mean(), val_df['v_x'].std()
    # val_v_y_mean, val_v_y_std = val_df['v_y'].mean(), val_df['v_y'].std()
    val_speed_mean, val_speed_std = val_df['speed'].mean(), val_df['speed'].std()
    val_heading_mean, val_heading_std = val_df['heading'].mean(), val_df['heading'].std()

    val_df['x_rel'] = (val_df['x_rel'] - val_x_mean) / val_x_std
    val_df['y_rel'] = (val_df['y_rel'] - val_y_mean) / val_y_std
    # val_df['rot_q0'] = (val_df['rot_q0'] - val_rot_q0_mean) / val_rot_q0_std
    # val_df['rot_q1'] = (val_df['rot_q1'] - val_rot_q1_mean) / val_rot_q1_std
    # val_df['rot_q2'] = (val_df['rot_q2'] - val_rot_q2_mean) / val_rot_q2_std
    # val_df['rot_q3'] = (val_df['rot_q3'] - val_rot_q3_mean) / val_rot_q3_std
    # val_df['v_x'] = (val_df['v_x'] - val_v_x_mean) / val_v_x_std
    # val_df['v_y'] = (val_df['v_y'] - val_v_y_mean) / val_v_y_std
    val_df['speed'] = (val_df['speed'] - val_speed_mean) / val_speed_std
    val_df['heading'] = (val_df['heading'] - val_heading_mean) / val_heading_std


    # describe the val data
    print(f'\nNORMALIZED VAL DATA')  
    print(val_df.describe() ) 

    print(f'\nTEST DATA')
    print(test_df.describe())
    
    # Save dataframes to csv files
    train_df.to_csv(os.path.join(train_dir, 'data.csv'), index=False)
    val_df.to_csv(os.path.join(val_dir, 'data.csv'), index=False)
    test_df.to_csv(os.path.join(test_dir, 'data.csv'), index=False)
    print(f'\nSAVED DATASETS')

    # Understand the train and validation data 
    print(f'\nTRAINING DATA')
    # create a histogram indicating how many instances are in each category
    train_category_counts = dict()
    train_category_counts[0] = len(train_df[train_df['category_0'] == 1])
    train_category_counts[1] = len(train_df[train_df['category_1'] == 1])
    # train_category_counts[2] = len(train_df[train_df['category_2'] == 1])
    # print("category percentage in training data:", train_category_counts[0] / len(train_df), train_category_counts[1] / len(train_df), train_category_counts[2] / len(train_df))
    print("category percentage in training data:", train_category_counts[0] / len(train_df), train_category_counts[1] / len(train_df))
    # plot the histogram
    plt.bar(train_category_counts.keys(), train_category_counts.values())
    plt.title('Training data category distribution')
    plt.xlabel('Category')
    plt.ylabel('Number of instances')
    plt.show()

    print(f'\nVALIDATION DATA')
    # create a histogram indicating how many instances are in each category
    val_category_counts = dict()
    val_category_counts[0] = len(val_df[val_df['category_0'] == 1])
    val_category_counts[1] = len(val_df[val_df['category_1'] == 1])
    # val_category_counts[2] = len(val_df[val_df['category_2'] == 1])
    # print("category percentage in validation data:", val_category_counts[0] / len(val_df), val_category_counts[1] / len(val_df), val_category_counts[2] / len(val_df))
    print("category percentage in validation data:", val_category_counts[0] / len(val_df), val_category_counts[1] / len(val_df))
    # plot the histogram
    plt.bar(val_category_counts.keys(), val_category_counts.values())
    plt.title('Validation data category distribution')
    plt.xlabel('Category')
    plt.ylabel('Number of instances')
    plt.show()
    print(f'\nTEST DATA')
    # create a histogram indicating how many instances are in each category
    test_category_counts = dict()
    test_category_counts[0] = len(test_df[test_df['category_0'] == 1])
    test_category_counts[1] = len(test_df[test_df['category_1'] == 1])
    # test_category_counts[2] = len(test_df[test_df['category_2'] == 1])
    # print("category percentage in test data:", test_category_counts[0] / len(test_df), test_category_counts[1] / len(test_df), test_category_counts[2] / len(test_df))
    print("category percentage in test data:", test_category_counts[0] / len(test_df), test_category_counts[1] / len(test_df))
    # plot the histogram
    plt.bar(test_category_counts.keys(), test_category_counts.values())
    plt.title('Test data category distribution')
    plt.xlabel('Category')
    plt.ylabel('Number of instances')
    plt.show()

    
    
    



       
