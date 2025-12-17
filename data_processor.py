import pandas as pd
import numpy as np
import os
import gc

# Configuration
TIME_BIN_SIZE = '1H'
START_DATE = '2025-07-01'
END_DATE = '2025-09-30'

def get_unique_locations(trips_path):
    """
    Scans the OD file to find all unique LocationIDs to define our Graph Nodes.
    """
    print("Scanning for unique Location IDs...")
    df = pd.read_parquet(trips_path, columns=['PULocationID', 'DOLocationID'])
    unique_pu = df['PULocationID'].unique()
    unique_do = df['DOLocationID'].unique()
    all_locs = np.unique(np.concatenate([unique_pu, unique_do]))

    # Filter to standard NYC Taxi Zones (1-263)
    all_locs = all_locs[(all_locs > 0) & (all_locs < 264)]

    print(f"Found {len(all_locs)} unique zones.")
    return sorted(all_locs)

def process_data(events_path, trips_path, output_dir='processed_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Define Nodes (Zones)
    valid_ids = get_unique_locations(trips_path)
    loc_to_idx = {loc: i for i, loc in enumerate(valid_ids)}
    num_nodes = len(valid_ids)
    np.save(f'{output_dir}/loc_mapping.npy', valid_ids)

    # 2. Define Time Bins
    time_bins = pd.date_range(start=START_DATE, end=END_DATE, freq=TIME_BIN_SIZE)
    num_timesteps = len(time_bins) - 1
    print(f"Time steps: {num_timesteps}, Nodes: {num_nodes}")

    # 3. Build FLOW Tensor (Feature 1) 
    print("Processing Flow Data...")

    # Initialize Flow Tensor: (Time, Nodes, 1)
    flow_tensor = np.zeros((num_timesteps, num_nodes, 1), dtype=np.float32)

    df = pd.read_parquet(trips_path, columns=['request_datetime', 'PULocationID'])

    # Bin times
    df['time_idx'] = df['request_datetime'].apply(lambda x: time_bins.searchsorted(x) - 1)
    df = df[(df['time_idx'] >= 0) & (df['time_idx'] < num_timesteps)]
    df = df[df['PULocationID'].isin(valid_ids)]

    # Map to Index
    df['node_idx'] = df['PULocationID'].map(loc_to_idx)

    # Aggregate counts (Demand per zone per hour)
    counts = df.groupby(['time_idx', 'node_idx']).size().reset_index(name='trips')

    # Fill Flow Tensor
    indices = counts[['time_idx', 'node_idx']].values.astype(int)
    values = counts['trips'].values.astype(np.float32)
    flow_tensor[indices[:,0], indices[:,1], 0] = values

    del df, counts
    gc.collect()

    # 4. Build event Tensor (Feature 2) 
    print("Processing Event Data...")
    events = pd.read_csv(events_path, parse_dates=['Start Date/Time', 'End Date/Time'])

    # Initialize Event Tensor: (Time, Nodes, 1)
    event_tensor = np.zeros((num_timesteps, num_nodes, 1), dtype=np.float32)

    for row in events.itertuples():
        loc_id = getattr(row, "LocationID") 

        if loc_id in loc_to_idx:
            idx = loc_to_idx[loc_id]


            s_time = max(row._3, pd.Timestamp(START_DATE))
            e_time = min(row._4, pd.Timestamp(END_DATE))

            if s_time < e_time:
                start_idx = max(0, time_bins.searchsorted(s_time) - 1)
                end_idx = min(num_timesteps, time_bins.searchsorted(e_time))

                # Mark 1.0 for presence of event
                if end_idx > start_idx:
                    event_tensor[start_idx:end_idx, idx, 0] = 1.0

    # 5. Merge and Save
    print("Merging Tensors...")

    # Final Shape: (Time, Nodes, Features)
    # Feature 0: Traffic Flow (Demand)
    # Feature 1: Event Presence (Binary 0/1)
    X = np.concatenate([flow_tensor, event_tensor], axis=2)

    print(f"Final Input Tensor Shape: {X.shape}")
    np.save(f'{output_dir}/X_input.npy', X)

    # Create Target (Y) - typically predicting next step Flow
    Y = flow_tensor[1:] 
    # Adjust X to match length of Y
    X = X[:-1]

    np.save(f'{output_dir}/Y_target.npy', Y)
    print("Processing Complete. Files saved to /processed_data")

if __name__ == "__main__":
    process_data('mapped_events.csv', 'manhattan_merged.parquet')
