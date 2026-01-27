import torch

from torch.utils.data import Dataset, DataLoader


def build_dataloader(
    args, 
    snapshots, 
    sketching_matrix=None, 
    drop_last=False, 
    shuffle=True, 
    batch_size=None
):
    """  Build dataloader for a single window 
    
    Args:
      args: arguments
      snapshots: snapshots
      sketching_matrix: sketching matrix
      drop_last: drop last batch
      shuffle: shuffle data
      batch_size: batch size
      
    Returns:
      dataloader: dataloader
      sketching_matrix: sketching matrix
      
    """
    if sketching_matrix is not None:
        sketching_matrix = torch.from_numpy(sketching_matrix).float()
        
    batch_size = args.batch_size if batch_size is None else batch_size
    dataloader = DataLoader(
            WindowDataset(snapshots),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    
    return dataloader, sketching_matrix


class WindowDataset(Dataset):
    """ Streaming dataset for a single window """
    def __init__(self, counters):
        super().__init__()
        self.sample_num = counters.shape[0]
        self.samples = counters

    def __getitem__(self, ind):
        return torch.from_numpy(self.samples[ind, :, :]).float()
    
    def __len__(self):
        return self.sample_num
