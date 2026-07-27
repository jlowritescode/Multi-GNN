import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
import numpy as np

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1) if not isinstance(data, HeteroData) else torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def to_adj_edges_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u,v,t) in enumerate(edges):
        u,v,t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def time_deltas(data, adj_edges_list):
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1: continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i+1,-1] - a[i,-1] for i in range(a.shape[0]-1)]
        tds = np.hstack((a[:,0].reshape(-1,1), np.array(a_tds).reshape(-1,1)))
        for i,td in tds:
            time_deltas[i] = td
    return time_deltas

def cross_direction_time_deltas(data):
    """
    Calculate causal cross-direction temporal features for every transaction.

    For edge u -> v at time t:

    0. Time since u last received a transaction.
    1. Whether u has previously received a transaction.
    2. Time since v last sent a transaction.
    3. Whether v has previously sent a transaction.

    Transactions sharing the exact same timestamp are processed together, so
    one same-time transaction is not treated as preceding another.
    """
    if data.timestamps is None:
        return torch.zeros(
            (data.edge_index.shape[1], 4),
            dtype=torch.float32,
        )

    edge_index = data.edge_index
    timestamps = data.timestamps.to(torch.float64)

    num_edges = edge_index.shape[1]
    num_nodes = data.num_nodes

    features = torch.zeros((num_edges, 4), dtype=torch.float32)

    last_incoming = torch.full(
        (num_nodes,),
        float("nan"),
        dtype=torch.float64,
    )
    last_outgoing = torch.full(
        (num_nodes,),
        float("nan"),
        dtype=torch.float64,
    )

    # Globally process transactions in chronological order.
    order = torch.argsort(timestamps, stable=True)
    sorted_times = timestamps[order]

    start = 0

    while start < num_edges:
        current_time = sorted_times[start]

        # Find all transactions with this exact timestamp.
        end = start + 1
        while end < num_edges and sorted_times[end] == current_time:
            end += 1

        edge_ids = order[start:end]

        sources = edge_index[0, edge_ids]
        destinations = edge_index[1, edge_ids]

        source_last_incoming = last_incoming[sources]
        destination_last_outgoing = last_outgoing[destinations]

        source_has_prior_incoming = ~torch.isnan(source_last_incoming)
        destination_has_prior_outgoing = ~torch.isnan(
            destination_last_outgoing
        )

        features[edge_ids, 0] = torch.where(
            source_has_prior_incoming,
            current_time - source_last_incoming,
            0.0,
        ).float()

        features[edge_ids, 1] = source_has_prior_incoming.float()

        features[edge_ids, 2] = torch.where(
            destination_has_prior_outgoing,
            current_time - destination_last_outgoing,
            0.0,
        ).float()

        features[edge_ids, 3] = destination_has_prior_outgoing.float()

        # Update history only after every edge at this timestamp was scored.
        last_outgoing[sources] = current_time
        last_incoming[destinations] = current_time

        start = end

    return features



class TemporalEventIndex:
    """
    Indexes transaction timestamps by account.

    After construction, all event timestamps are sorted first by account ID
    and then by timestamp. This supports efficient causal rolling counts.
    """

    def __init__(
        self,
        event_nodes: np.ndarray,
        event_times: np.ndarray,
        num_nodes: int,
    ):
        event_nodes = np.asarray(event_nodes, dtype=np.int64)
        event_times = np.asarray(event_times, dtype=np.int64)

        if event_nodes.ndim != 1 or event_times.ndim != 1:
            raise ValueError("event_nodes and event_times must be 1D arrays.")

        if len(event_nodes) != len(event_times):
            raise ValueError(
                "event_nodes and event_times must have equal lengths."
            )

        self.num_nodes = int(num_nodes)

        # np.lexsort uses the final key as the primary key:
        # first sort by node, then by timestamp within each node.
        order = np.lexsort((event_times, event_nodes))

        sorted_nodes = event_nodes[order]
        self.sorted_times = event_times[order]

        # offsets[node] : offsets[node + 1] gives that node's timestamps.
        node_counts = np.bincount(
            sorted_nodes,
            minlength=self.num_nodes,
        )

        self.offsets = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.cumsum(node_counts, out=self.offsets[1:])

    def write_rolling_counts(
        self,
        query_nodes: np.ndarray,
        query_times: np.ndarray,
        windows: tuple[int, ...],
        output: np.ndarray,
        column_offset: int,
    ) -> None:
        """
        Write counts of historical events in [query_time - window, query_time).

        The right side is open, so transactions at exactly query_time are not
        considered historical events.
        """
        query_nodes = np.asarray(query_nodes, dtype=np.int64)
        query_times = np.asarray(query_times, dtype=np.int64)

        if query_nodes.shape != query_times.shape:
            raise ValueError(
                "query_nodes and query_times must have identical shapes."
            )

        # Group the queries by account so each account's history is retrieved
        # only once.
        query_order = np.argsort(query_nodes, kind="stable")
        sorted_query_nodes = query_nodes[query_order]

        unique_nodes, group_starts = np.unique(
            sorted_query_nodes,
            return_index=True,
        )

        group_ends = np.concatenate(
            [
                group_starts[1:],
                np.array([len(query_order)], dtype=np.int64),
            ]
        )

        for node, group_start, group_end in zip(
            unique_nodes,
            group_starts,
            group_ends,
        ):
            if node < 0 or node >= self.num_nodes:
                raise IndexError(f"Account ID {node} is out of bounds.")

            history_start = self.offsets[node]
            history_end = self.offsets[node + 1]

            if history_start == history_end:
                continue

            history_times = self.sorted_times[
                history_start:history_end
            ]

            query_indices = query_order[group_start:group_end]
            times = query_times[query_indices]

            # Strictly exclude transactions at the target timestamp.
            right_boundaries = np.searchsorted(
                history_times,
                times,
                side="left",
            )

            for window_index, window_seconds in enumerate(windows):
                left_boundaries = np.searchsorted(
                    history_times,
                    times - window_seconds,
                    side="left",
                )

                counts = right_boundaries - left_boundaries

                output[
                    query_indices,
                    column_offset + window_index,
                ] = counts


def rolling_transaction_velocity(
    edge_index: torch.Tensor,
    timestamps: torch.Tensor,
    num_nodes: int,
    windows: tuple[int, ...] = (600, 3600, 86400),
    log_transform: bool = True,
) -> torch.Tensor:
    """
    Calculate causal rolling transaction-count features.

    For transaction u -> v at time t, calculates:

        u's previous outgoing count
        u's previous incoming count
        v's previous outgoing count
        v's previous incoming count

    for every specified rolling window.

    Parameters
    ----------
    edge_index:
        Tensor with shape [2, num_edges].
    timestamps:
        Tensor with shape [num_edges].
    num_nodes:
        Number of account nodes.
    windows:
        Rolling windows in seconds.
    log_transform:
        Apply log(1 + count) to reduce count skew.

    Returns
    -------
    Tensor with shape:

        [num_edges, 4 * len(windows)]

    Feature order is role-major:

        src_out_window_1, ..., src_out_window_n,
        src_in_window_1,  ..., src_in_window_n,
        dst_out_window_1, ..., dst_out_window_n,
        dst_in_window_1,  ..., dst_in_window_n
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    windows = tuple(int(window) for window in windows)

    if not windows:
        raise ValueError("At least one rolling window is required.")

    if any(window <= 0 for window in windows):
        raise ValueError("Every rolling window must be positive.")

    source_nodes = (
        edge_index[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    destination_nodes = (
        edge_index[1]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    # Timestamps are stored in seconds in this repository.
    event_times = (
        timestamps
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    num_edges = edge_index.shape[1]
    num_windows = len(windows)

    features = np.zeros(
        (num_edges, 4 * num_windows),
        dtype=np.float32,
    )

    # Index outgoing transaction history by sender.
    outgoing_index = TemporalEventIndex(
        event_nodes=source_nodes,
        event_times=event_times,
        num_nodes=num_nodes,
    )

    # Source account's prior outgoing activity.
    outgoing_index.write_rolling_counts(
        query_nodes=source_nodes,
        query_times=event_times,
        windows=windows,
        output=features,
        column_offset=0,
    )

    # Destination account's prior outgoing activity.
    outgoing_index.write_rolling_counts(
        query_nodes=destination_nodes,
        query_times=event_times,
        windows=windows,
        output=features,
        column_offset=2 * num_windows,
    )

    # Release the outgoing index before constructing the incoming index.
    # This reduces peak memory usage on large datasets.
    del outgoing_index

    # Index incoming transaction history by receiver.
    incoming_index = TemporalEventIndex(
        event_nodes=destination_nodes,
        event_times=event_times,
        num_nodes=num_nodes,
    )

    # Source account's prior incoming activity.
    incoming_index.write_rolling_counts(
        query_nodes=source_nodes,
        query_times=event_times,
        windows=windows,
        output=features,
        column_offset=num_windows,
    )

    # Destination account's prior incoming activity.
    incoming_index.write_rolling_counts(
        query_nodes=destination_nodes,
        query_times=event_times,
        windows=windows,
        output=features,
        column_offset=3 * num_windows,
    )

    del incoming_index

    if log_transform:
        # Counts are often highly right-skewed. log1p preserves zero while
        # preventing extremely active accounts from dominating the scale.
        np.log1p(features, out=features)

    return torch.from_numpy(features)


def rolling_velocity_feature_names(
    windows: tuple[int, ...],
) -> list[str]:
    """Return names in the same order as rolling_transaction_velocity."""
    roles = (
        "src_out",
        "src_in",
        "dst_out",
        "dst_in",
    )

    return [
        f"{role}_count_{window}s"
        for role in roles
        for window in windows
    ]
class GraphData(Data):
    '''This is the homogenous graph object we use for GNN training if reverse MP is not enabled'''
    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, 
        readout: str = 'edge', 
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
        ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps  
        elif edge_attr is not None:
            self.timestamps = edge_attr[:,0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = [ports(self.edge_index.flipud(), adj_list_out)] if reverse_ports else []
        self.edge_attr = torch.cat([self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self

    def add_cross_direction_time_deltas(self):
        """
        Add causal pass-through timing features to transaction edge attributes.
        """
        flow_tds = cross_direction_time_deltas(self)

        self.edge_attr = torch.cat(
            [self.edge_attr, flow_tds],
            dim=1,
        )

        return self

class HeteroGraphData(HeteroData):
    '''This is the heterogenous graph object we use for GNN training if reverse MP is enabled'''
    def __init__(
        self,
        readout: str = 'edge',
        **kwargs
        ):
        super().__init__(**kwargs)
        self.readout = readout

    @property
    def num_nodes(self):
        return self['node'].x.shape[0]
        
    @property
    def timestamps(self):
        return self['node', 'to', 'node'].timestamps

    def add_ports(self):
        '''Adds port numberings to the edge features'''
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self['node', 'to', 'node'].edge_index, adj_list_in)
        out_ports = ports(self['node', 'rev_to', 'node'].edge_index, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_ports], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1)
        return self

    def add_time_deltas(self):
        '''Adds time deltas (i.e. the time between subsequent transactions) to the edge features'''
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = time_deltas(self, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_tds], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_tds], dim=1)
        return self
    
def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def create_hetero_obj(x,  y,  edge_index,  edge_attr, timestamps, args):
    '''Creates a heterogenous graph object for reverse message passing'''
    data = HeteroGraphData()

    data['node'].x = x
    data['node', 'to', 'node'].edge_index = edge_index
    data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
    data['node', 'to', 'node'].edge_attr = edge_attr
    data['node', 'rev_to', 'node'].edge_attr = edge_attr
    if args.ports:
        #swap the in- and outgoing port numberings for the reverse edges
        data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]
    data['node', 'to', 'node'].y = y
    data['node', 'to', 'node'].timestamps = timestamps
    
    return data