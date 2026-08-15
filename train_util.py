import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score
import json

class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

def extract_param(parameter_name: str, args) -> float:
    """
    Extract the value of the specified parameter for the given model.
    
    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.
    
    Returns:
    - float: Value of the specified parameter.
    """
    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)

def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y


        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), tr_edge_label_index), 
                                    edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=True, transform=transform)
        
        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:,val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]


        val_loader =  LinkNeighborLoader(val_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), val_edge_label_index), 
                                    edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
        
        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:,te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]


        te_loader =  LinkNeighborLoader(te_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), te_edge_label_index), 
                                    edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
    else:
        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True, transform=transform)
        val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        
    return tr_loader, val_loader, te_loader


@torch.no_grad()
def evaluate_homo(
    loader,
    inds,
    model,
    data,
    device,
    args,
    return_details=False
):
    """
    Evaluates model performance for homogeneous graph data.

    Normally returns only F1, preserving the repo's existing behavior.

    If return_details=True, also returns:
        - ground-truth labels
        - predicted labels
        - laundering probabilities
    """

    preds = []
    ground_truths = []
    scores = []
    edge_ids = []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):

        # Select the seed edges from which the batch was created
        inds_cpu = inds.detach().cpu()

        batch_edge_inds = inds_cpu[
            batch.input_id.detach().cpu()
        ]

        batch_edge_ids = (
            loader.data.edge_attr
            .detach()
            .cpu()[batch_edge_inds, 0]
        )

        mask = torch.isin(
            batch.edge_attr[:, 0].detach().cpu(),
            batch_edge_ids
        )

        # Add seed edges that were not sampled into the batch
        missing = ~torch.isin(
            batch_edge_ids,
            batch.edge_attr[:, 0].detach().cpu()
        )

        if (
            missing.sum() != 0
            and (args.data == 'Small_J' or args.data == 'Small_Q')
        ):

            missing_ids = batch_edge_ids[missing].int()

            n_ids = batch.n_id

            add_edge_index = (
                data.edge_index[:, missing_ids]
                .detach()
                .clone()
            )

            node_mapping = {
                value.item(): idx
                for idx, value in enumerate(n_ids)
            }

            add_edge_index = torch.tensor([
                [
                    node_mapping[val.item()]
                    for val in row
                ]
                for row in add_edge_index
            ])

            add_edge_attr = (
                data.edge_attr[missing_ids, :]
                .detach()
                .clone()
            )

            add_y = (
                data.y[missing_ids]
                .detach()
                .clone()
            )

            batch.edge_index = torch.cat(
                (batch.edge_index, add_edge_index),
                1
            )

            batch.edge_attr = torch.cat(
                (batch.edge_attr, add_edge_attr),
                0
            )

            batch.y = torch.cat(
                (batch.y, add_y),
                0
            )

            mask = torch.cat(
                (
                    mask,
                    torch.ones(
                        add_y.shape[0],
                        dtype=torch.bool
                    )
                )
            )

        # Remove unique edge ID
        # Save the IDs corresponding exactly to the evaluated edges.
        # These IDs are aligned with out[mask].
        current_edge_ids = (
            batch.edge_attr[mask, 0]
            .detach()
            .cpu()
        )

        edge_ids.append(current_edge_ids)
        batch.edge_attr = batch.edge_attr[:, 1:]

        batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr
        )

        # Move mask to same device as model output
        mask = mask.to(out.device)

        out = out[mask]

        # Standard class prediction
        pred = out.argmax(dim=-1)

        # Probability of laundering class = class 1
        probability = torch.softmax(
            out,
            dim=-1
        )[:, 1]

        preds.append(pred)
        scores.append(probability)
        ground_truths.append(batch.y[mask])

    pred = (
        torch.cat(preds, dim=0)
        .cpu()
        .numpy()
    )

    ground_truth = (
        torch.cat(ground_truths, dim=0)
        .cpu()
        .numpy()
    )

    score = (
        torch.cat(scores, dim=0)
        .cpu()
        .numpy()
    )

    f1 = f1_score(
        ground_truth,
        pred,
        zero_division=0
    )

    # Existing training behavior
    if not return_details:
        return f1

    # Extra information needed for final reporting

    edge_id = (
        torch.cat(edge_ids, dim=0)
        .cpu()
        .numpy()
        .astype(int)
    )
    return {
        "f1": f1,
        "y_true": ground_truth,
        "y_pred": pred,
        "y_score": score,
        "edge_ids": edge_id
    }

@torch.no_grad()
def evaluate_hetero(
    loader,
    inds,
    model,
    data,
    device,
    args,
    return_details=False
):
    """
    Evaluates model performance for heterogeneous graph data.

    Normally returns only F1.

    If return_details=True, also returns ground truth,
    predictions, and laundering probabilities.
    """
    preds = []
    ground_truths = []
    scores = []
    edge_ids = []

    for batch in tqdm.tqdm(loader, disable=not args.tqdm):

        # Select the seed edges from which the batch was created
        inds_cpu = inds.detach().cpu()

        batch_edge_inds = inds_cpu[
            batch[
                'node',
                'to',
                'node'
            ].input_id.detach().cpu()
        ]

        batch_edge_ids = (
            loader.data[
                'node',
                'to',
                'node'
            ].edge_attr
            .detach()
            .cpu()[batch_edge_inds, 0]
        )

        mask = torch.isin(
            batch[
                'node',
                'to',
                'node'
            ].edge_attr[:, 0].detach().cpu(),
            batch_edge_ids
        )

        # Add seed edges that were not sampled
        missing = ~torch.isin(
            batch_edge_ids,
            batch[
                'node',
                'to',
                'node'
            ].edge_attr[:, 0].detach().cpu()
        )

        if (
            missing.sum() != 0
            and (args.data == 'Small_J' or args.data == 'Small_Q')
        ):

            missing_ids = batch_edge_ids[missing].int()

            n_ids = batch['node'].n_id

            add_edge_index = (
                data[
                    'node',
                    'to',
                    'node'
                ].edge_index[:, missing_ids]
                .detach()
                .clone()
            )

            node_mapping = {
                value.item(): idx
                for idx, value in enumerate(n_ids)
            }

            add_edge_index = torch.tensor([
                [
                    node_mapping[val.item()]
                    for val in row
                ]
                for row in add_edge_index
            ])

            add_edge_attr = (
                data[
                    'node',
                    'to',
                    'node'
                ].edge_attr[missing_ids, :]
                .detach()
                .clone()
            )

            add_y = (
                data[
                    'node',
                    'to',
                    'node'
                ].y[missing_ids]
                .detach()
                .clone()
            )

            batch[
                'node',
                'to',
                'node'
            ].edge_index = torch.cat(
                (
                    batch[
                        'node',
                        'to',
                        'node'
                    ].edge_index,
                    add_edge_index
                ),
                1
            )

            batch[
                'node',
                'to',
                'node'
            ].edge_attr = torch.cat(
                (
                    batch[
                        'node',
                        'to',
                        'node'
                    ].edge_attr,
                    add_edge_attr
                ),
                0
            )

            batch[
                'node',
                'to',
                'node'
            ].y = torch.cat(
                (
                    batch[
                        'node',
                        'to',
                        'node'
                    ].y,
                    add_y
                ),
                0
            )

            mask = torch.cat(
                (
                    mask,
                    torch.ones(
                        add_y.shape[0],
                        dtype=torch.bool
                    )
                )
            )
        current_edge_ids = (
            batch[
                'node',
                'to',
                'node'
            ].edge_attr[mask, 0]
            .detach()
            .cpu()
        )

        edge_ids.append(current_edge_ids)
        # Remove unique edge IDs
        batch[
            'node',
            'to',
            'node'
        ].edge_attr = batch[
            'node',
            'to',
            'node'
        ].edge_attr[:, 1:]

        batch[
            'node',
            'rev_to',
            'node'
        ].edge_attr = batch[
            'node',
            'rev_to',
            'node'
        ].edge_attr[:, 1:]

        batch.to(device)

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict
        )

        out = out[
            ('node', 'to', 'node')
        ]

        # Move mask to same device
        mask = mask.to(out.device)

        out = out[mask]

        pred = out.argmax(dim=-1)

        # Probability of laundering class
        probability = torch.softmax(
            out,
            dim=-1
        )[:, 1]

        preds.append(pred)
        scores.append(probability)

        ground_truths.append(
            batch[
                'node',
                'to',
                'node'
            ].y[mask]
        )

    pred = (
        torch.cat(preds, dim=0)
        .cpu()
        .numpy()
    )

    ground_truth = (
        torch.cat(ground_truths, dim=0)
        .cpu()
        .numpy()
    )

    score = (
        torch.cat(scores, dim=0)
        .cpu()
        .numpy()
    )

    f1 = f1_score(
        ground_truth,
        pred,
        zero_division=0
    )
    edge_id = (
        torch.cat(edge_ids, dim=0)
        .cpu()
        .numpy()
        .astype(int)
    )
    if not return_details:
        return f1

    return {
        "f1": f1,
        "y_true": ground_truth,
        "y_pred": pred,
        "y_score": score
    }

def save_model(model, optimizer, epoch, args, data_config):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{"" if not args.finetune else "_finetuned"}.tar')
    
def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer