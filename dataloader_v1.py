import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch


def collate_fn(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = Batch.from_data_list(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


def build_dataloader(dataset_config):
    dataset = TUDataset(root=dataset_config['dataset_root'], name=dataset_config['dataset_name'])
    train_idx = torch.arange(len(dataset))
    train_sampler = SubsetRandomSampler(train_idx)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, collate_fn=collate_fn,
               batch_size=dataset_config['batch_size'], pin_memory=True)
    eval_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=dataset_config['batch_size'], shuffle=False)
    dataset_information = {
        'node_num': dataset.data.x.shape[0],
        'node_feat': dataset.data.x.shape[1],
        'edge_num': dataset.data.edge_index.shape[-1],
        # 'edge_feat': dataset.data.edge_attr.shape[-1],
        'graph_num': dataset.data.y.shape[0],
    }
    return train_dataloader, eval_dataloader, dataset_information


dataset_config = {
    'dataset_root': '/Users/yaoyuxiang/Documents/Code/PyCharmProjects/GraphMAE_PyG/data',
    'dataset_name': 'MUTAG',
    'batch_size': 10
}