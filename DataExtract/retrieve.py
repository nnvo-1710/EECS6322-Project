import h5py
import numpy as np
import torch
import pickle
from DataExtract.data_utils import *
from src import utils


# --- Normalization Methods ---
def normalize_eeg_data(eeg_clip, normalization_method="global"):
    """
    Normalize EEG data based on the specified method.
    
    Args:
        eeg_clip: The EEG clip data, shape (seq_len, num_nodes, input_dim)
        normalization_method: The normalization method to apply. Options: 
            'global', 'channel', 'clip', 'zscore'
    
    Returns:
        The normalized EEG data.
    """
    # Determine the method of normalization
    if normalization_method == "global":
        # Global normalization across all data (mean and std across all clips)
        mean = np.mean(eeg_clip)
        std = np.std(eeg_clip)
        normalized_eeg = (eeg_clip - mean) / std
    
    elif normalization_method == "channel":
        # Channel-wise normalization (mean and std per channel)
        mean = np.mean(eeg_clip, axis=(0, 1), keepdims=True)  # Mean over time and input_dim
        std = np.std(eeg_clip, axis=(0, 1), keepdims=True)  # Std over time and input_dim
        normalized_eeg = (eeg_clip - mean) / std
    
    elif normalization_method == "clip":
        # Clip-wise normalization (mean and std per clip)
        mean = np.mean(eeg_clip, axis=(0, 2), keepdims=True)  # Mean over nodes and input_dim
        std = np.std(eeg_clip, axis=(0, 2), keepdims=True)  # Std over nodes and input_dim
        normalized_eeg = (eeg_clip - mean) / std
    
    elif normalization_method == "zscore":
        # Z-score normalization (using z-score formula for each clip)
        mean = np.mean(eeg_clip, axis=(0, 1, 2), keepdims=True)
        std = np.std(eeg_clip, axis=(0, 1, 2), keepdims=True)
        normalized_eeg = (eeg_clip - mean) / std
    
    else:
        raise ValueError("Unknown normalization method: {}".format(normalization_method))
    
    return normalized_eeg

# --- Graph Computation Methods ---
def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_mx).tocoo()

def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    return (2 / lambda_max * L) - I


def get_indiv_graphs(eeg_clip, num_sensors, top_k):
    adj_mat = np.eye(num_sensors, dtype=np.float32)
    eeg_clip = np.transpose(eeg_clip, (1, 0, 2)).reshape((num_sensors, -1))
    
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            xcorr = comp_xcorr(eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
            adj_mat[i, j] = adj_mat[j, i] = abs(xcorr)
    
    if top_k is not None:
        adj_mat = keep_topk(adj_mat, top_k=top_k, directed=True)
    else:
        raise ValueError("Invalid top_k value!")
    
    return adj_mat

def get_combined_graph(adj_mat_dir, swap_nodes=None):
    """
    Get adjacency matrix for pre-computed distance graph
    Returns:
        adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
    """
    with open(adj_mat_dir, 'rb') as pf:
        adj_mat = pickle.load(pf)
        adj_mat = adj_mat[-1]

    adj_mat_new = adj_mat.copy()
    if swap_nodes is not None:
        for node_pair in swap_nodes:
            for i in range(adj_mat.shape[0]):
                adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                adj_mat_new[i, i] = 1
            adj_mat_new[node_pair[0], node_pair[1]] = adj_mat[node_pair[1], node_pair[0]]
            adj_mat_new[node_pair[1], node_pair[0]] = adj_mat[node_pair[0], node_pair[1]]

    return adj_mat_new

def compute_supports(adj_mat, filter_type="laplacian"):
    """
    Compute supports based on the adjacency matrix.
    Args:
        adj_mat: adjacency matrix
        filter_type: string specifying the filter type ('laplacian', 'random_walk', etc.)
    Returns:
        supports: list of support matrices (using different graph convolution types)
    """
    supports = []
    supports_mat = []

    if filter_type == "laplacian":  # ChebNet graph conv
        supports_mat.append(utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
    elif filter_type == "random_walk":  # Forward random walk
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
    elif filter_type == "dual_random_walk":  # Bidirectional random walk
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(utils.calculate_scaled_laplacian(adj_mat))

    for support in supports_mat:
        supports.append(torch.FloatTensor(support.toarray()))

    return supports

def load_all_data(directory, top_k=None, adj_mat_dir=None, use_combined_graph=False, filter_type="laplacian", padding_value=0.0):
    """Load all EEG clips and labels from files in the directory.

    Args:
        directory (str): Path to the directory containing .h5 files.
        top_k (int, optional): Number of strongest connections to keep in adjacency matrix.
        adj_mat_dir (str, optional): Path to precomputed adjacency matrix file.
        use_combined_graph (bool): If True, use precomputed adjacency matrix.
        filter_type (str): Type of filter for computing graph supports.
        padding_value (float): Value used for padding if required.

    Returns:
        clips (list): List of padded EEG clips.
        labels (list): List of labels corresponding to the clips.
        adj_mats (list): List of adjacency matrices corresponding to the clips.
        supports (list): List of computed supports for each clip.
    """
    clips, labels, adj_mats, supports = [], [], [], []

    files = [f for f in os.listdir(directory) if f.endswith('.h5')]

    for file in files:
        file_path = os.path.join(directory, file)
        with h5py.File(file_path, 'r') as f:
            eeg_clip = f['clip'][:]  # (seq_len, num_sensors, 100)
            label = int(f['label'][()])  # Extract scalar label

            num_sensors = eeg_clip.shape[1]  # Deriving num_sensors from the eeg_clip shape

            # Pad the sequence dimension (eeg_clip.shape[0]) if needed
            if eeg_clip.shape[0] < 12:  # Assuming the target sequence length is 12
                pad_width = ((12 - eeg_clip.shape[0], 0), (0, 0), (0, 0))  # pad the sequence dimension (first dimension)
                eeg_clip = np.pad(eeg_clip, pad_width, constant_values=padding_value)

            # Ensure that all clips are consistently the same shape (12, 19, 100)
            if eeg_clip.shape != (12, 19, 100):
                raise ValueError(f"Inconsistent shape detected: {eeg_clip.shape}")

            # Compute adjacency matrix
            if use_combined_graph and adj_mat_dir:
                adj_mat = get_combined_graph(adj_mat_dir)
            elif top_k is not None:
                adj_mat = get_indiv_graphs(eeg_clip, num_sensors, top_k)
            else:
                adj_mat = np.eye(num_sensors, dtype=np.float32)  # Identity if no graph

            # Compute supports
            support = compute_supports(adj_mat, filter_type) if adj_mat is not None else []

            clips.append(eeg_clip)
            labels.append(label)
            adj_mats.append(adj_mat)
            supports.append(support)

    return clips, labels, adj_mats, supports


def batch_from_data(clips, labels, adj_mats, supports, batch_size, filter_type="laplacian"):
    """Batch generator that returns batches from pre-loaded data.

    Args:
        clips (list): List of EEG clips (each of shape (12, 19, 100)).
        labels (list): List of labels corresponding to the clips.
        adj_mats (list): List of adjacency matrices corresponding to the clips.
        supports (list): List of supports corresponding to the clips.
        batch_size (int): Number of samples per batch.
        filter_type (str): Type of filter for computing graph supports.

    Yields:
        batch_clips: (batch_size, 12, 19, 100)
        batch_labels: (batch_size,)
        batch_adj_mats: (batch_size, num_sensors, num_sensors)
        batch_supports: List of support tensors per sample.
    """
    num_samples = len(clips)
    indices = np.arange(num_samples)

    # Shuffle data if necessary
    np.random.shuffle(indices)

    batch_clips, batch_labels, batch_adj_mats, batch_supports = [], [], [], []

    for idx in indices:
        batch_clips.append(clips[idx])
        batch_labels.append(labels[idx])
        batch_adj_mats.append(adj_mats[idx])
        batch_supports.append(supports[idx])

        # Yield the batch when it's full
        if len(batch_clips) == batch_size:
            batch_clips_np = np.stack(batch_clips)
            yield (
                batch_clips_np, 
                np.array(batch_labels), 
                np.stack(batch_adj_mats), 
                [torch.FloatTensor(support) if isinstance(support, np.ndarray) else support for support in batch_supports]
            )
            batch_clips, batch_labels, batch_adj_mats, batch_supports = [], [], [], []

    # Yield the remaining data
    if batch_clips:
        batch_clips_np = np.stack(batch_clips)
        yield (
            batch_clips_np, 
            np.array(batch_labels), 
            np.stack(batch_adj_mats), 
            [torch.FloatTensor(support) if isinstance(support, np.ndarray) else support for support in batch_supports]
        )




def batch_generator_from_files(directory, batch_size, top_k=None, adj_mat_dir=None, shuffle_data=True, filter_type="laplacian", use_combined_graph=False, padding_value=0.0):
    """Batch generator that reads EEG clips in batches directly from files.

    Args:
        directory (str): Path to the directory containing .h5 files.
        batch_size (int): Number of samples per batch.
        top_k (int, optional): Number of strongest connections to keep in adjacency matrix.
        adj_mat_dir (str, optional): Path to precomputed adjacency matrix file.
        shuffle_data (bool): Whether to shuffle the order of files before reading.
        filter_type (str): Type of filter for computing graph supports.
        use_combined_graph (bool): If True, use precomputed adjacency matrix.
        padding_value (float): Value used for padding if required.

    Yields:
        batch_clips: (batch_size, 12, 19, 100)
        batch_labels: (batch_size,)
        batch_adj_mats: (batch_size, num_sensors, num_sensors) or None
        batch_supports: List of support tensors per sample.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.h5')]

    if shuffle_data:
        np.random.shuffle(files)  # Shuffle files if enabled

    batch_clips, batch_labels, batch_adj_mats, batch_supports = [], [], [], []

    for file in files:
        file_path = os.path.join(directory, file)
        with h5py.File(file_path, 'r') as f:
            eeg_clip = f['clip'][:]  # (seq_len, num_sensors, 100)
            label = int(f['label'][()])  # Extract scalar label

            num_sensors = eeg_clip.shape[1]  # Deriving num_sensors from the eeg_clip shape

            # Pad the sequence dimension (eeg_clip.shape[0]) if needed
            if eeg_clip.shape[0] < 12:  # Assuming the target sequence length is 12
                pad_width = ((12 - eeg_clip.shape[0], 0), (0, 0), (0, 0))  # pad the sequence dimension (first dimension)
                eeg_clip = np.pad(eeg_clip, pad_width, constant_values=padding_value)

            # Ensure that all clips are consistently the same shape (12, 19, 100)
            if eeg_clip.shape != (12, 19, 100):
                print(f"Padding applied to clip with shape {eeg_clip.shape}")
                # You can choose to handle this case differently, but this will raise an issue
                # if inconsistent shapes are found.
                raise ValueError(f"Inconsistent shape detected: {eeg_clip.shape}")

            # Compute adjacency matrix
            if use_combined_graph and adj_mat_dir:
                adj_mat = get_combined_graph(adj_mat_dir)
            elif top_k is not None:
                adj_mat = get_indiv_graphs(eeg_clip, num_sensors, top_k)
            else:
                adj_mat = np.eye(num_sensors, dtype=np.float32)  # Identity if no graph

            # Compute supports
            support = compute_supports(adj_mat, filter_type) if adj_mat is not None else []

            # Convert supports to numpy arrays or tensors (if needed)
            batch_clips.append(eeg_clip)
            batch_labels.append(label)
            batch_adj_mats.append(adj_mat)
            batch_supports.append(support)

        # Yield when batch is full
        if len(batch_clips) == batch_size:
            try:
                # Ensure all clips in the batch have the same shape before stacking
                batch_clips_np = np.stack(batch_clips)
                yield (
                    batch_clips_np, 
                    np.array(batch_labels), 
                    np.stack(batch_adj_mats) if batch_adj_mats else None, 
                    [torch.FloatTensor(support) if isinstance(support, np.ndarray) else support for support in batch_supports]  # Convert support if needed
                )
            except ValueError as e:
                print(f"Error stacking batch: {e}")
                for clip in batch_clips:
                    print(f"Clip shape: {clip.shape}")
                raise e  # Re-raise the error after printing the debug information

            # Reset batch lists
            batch_clips, batch_labels, batch_adj_mats, batch_supports = [], [], [], []

    # Yield any remaining data
    if batch_clips:
        try:
            batch_clips_np = np.stack(batch_clips)
            yield (
                batch_clips_np, 
                np.array(batch_labels), 
                np.stack(batch_adj_mats) if batch_adj_mats else None, 
                [torch.FloatTensor(support) if isinstance(support, np.ndarray) else support for support in batch_supports]  # Convert support if needed
            )
        except ValueError as e:
            print(f"Error stacking remaining batch: {e}")
            for clip in batch_clips:
                print(f"Clip shape: {clip.shape}")
            raise e  # Re-raise the error after printing the debug information






# Test the batch generator
def test_batch_generator():
    batch_gen = batch_generator_from_files(directory=directory, 
                                            batch_size=batch_size, 
                                            shuffle_data=True,
					    top_k=5,
                                            filter_type=filter_type, 
                                            use_combined_graph=use_combined_graph)
    
    for batch_idx, (batch_clips, batch_labels, batch_adj_mats, batch_supports) in enumerate(batch_gen):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Batch Clips Shape: {batch_clips.shape}")
        print(f"  Batch Labels Shape: {batch_labels.shape}")
        print(f"  Batch Adjacency Matrices Shape: {batch_adj_mats.shape}")
        print(f"  Labels: {batch_labels}")
        print("-" * 60)
        print(batch_adj_mats[1])
        # Break after processing one batch (for testing purposes)
        break  # Remove or adjust this line to process more batches

def test_load_all_data(directory, top_k=None, adj_mat_dir=None, use_combined_graph=False, filter_type="laplacian", padding_value=0.0):
    """
    Test function for loading EEG data and reporting label distribution.

    Args:
        directory (str): Path to the directory containing .h5 files.
        top_k (int, optional): Number of strongest connections to keep in adjacency matrix.
        adj_mat_dir (str, optional): Path to precomputed adjacency matrix file.
        use_combined_graph (bool): If True, use precomputed adjacency matrix.
        filter_type (str): Type of filter for computing graph supports.
        padding_value (float): Value used for padding if required.

    Prints:
        Number of clips loaded.
        Label distribution for multi-class labels (0 to 3).
    """
    clips, labels, adj_mats, supports = load_all_data(
        directory,
        top_k=top_k,
        adj_mat_dir=adj_mat_dir,
        use_combined_graph=use_combined_graph,
        filter_type=filter_type,
        padding_value=padding_value
    )

    print(f"Loaded {len(clips)} EEG clips.")

    # Calculate label distribution for multi-class labels (0 to 3)
    total = len(labels)
    label_counts = [0, 0, 0, 0]  # Assuming labels are between 0 and 3

    for label in labels:
        if 0 <= label <= 3:
            label_counts[label] += 1

    print("Label distribution:")
    for i in range(4):
        print(f"  Label {i}: {label_counts[i]} ({(label_counts[i] / total) * 100:.2f}%)")

    return clips, labels, adj_mats, supports

def test_batch_from_data():
    """Test the batch_from_data function."""
    # Example data
    clips = np.random.rand(100, 12, 19, 100)  # 100 samples, padded to shape (12, 19, 100)
    labels = np.random.randint(0, 2, size=(100,))  # Random labels (binary classification)
    adj_mats = np.random.rand(100, 19, 19)  # Random adjacency matrices for 100 samples
    supports = [np.random.rand(19, 19) for _ in range(100)]  # Random supports for each sample

    batch_size = 10  # Test with batch size 10
    filter_type = 'laplacian'  # Example filter type

    batch_gen = batch_from_data(clips, labels, adj_mats, supports, batch_size, filter_type)

    for batch_idx, (batch_clips, batch_labels, batch_adj_mats, batch_supports) in enumerate(batch_gen):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Batch Clips Shape: {batch_clips.shape}")
        print(f"  Batch Labels Shape: {batch_labels.shape}")
        print(f"  Batch Adjacency Matrices Shape: {batch_adj_mats.shape}")
        print(f"  Batch Supports Length: {len(batch_supports)}")

        # Check that each batch is the expected shape
        assert batch_clips.shape == (batch_size, 12, 19, 100), f"Unexpected batch clips shape: {batch_clips.shape}"
        assert batch_labels.shape == (batch_size,), f"Unexpected batch labels shape: {batch_labels.shape}"
        assert batch_adj_mats.shape == (batch_size, 19, 19), f"Unexpected batch adjacency matrix shape: {batch_adj_mats.shape}"
        
        # Check that supports are returned correctly
        for support in batch_supports:
            assert isinstance(support, torch.Tensor), f"Unexpected support type: {type(support)}"
            assert support.shape == (19, 19), f"Unexpected support shape: {support.shape}"

        if batch_idx == 2:  # Limit the number of batches for testing
            break

    print("batch_from_data test passed.")

# Run tests
directory = "./detection_pre/train/clipLen12_timeStepSize1"
test_load_all_data(directory)


