import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from functools import partial
from model_sub_v1 import GAT, GCN, GIN
from torch_geometric.data import Data
import copy


# loss function: sce
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


# loss function: sig
def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


# graph transformation: drop edge
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = Data(edge_index=torch.concat((nsrc, ndst), 0))
    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def setup_model(model_config):
    m_type = model_config['m_type']
    if m_type == "gat":
        model = GAT(model_config)
    elif m_type == "gin":
        model = GIN(model_config)
    elif m_type == "gcn":
        model = GCN(model_config)
    elif m_type == "mlp":
        model = nn.Sequential(
            nn.Linear(model_config['in_dim'], model_config['num_hidden']),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_config['num_hidden'], model_config['out_dim'])
        )
    elif m_type == "linear":
        model = nn.Linear(model_config['in_dim'], model_config['out_dim'])
    else:
        raise NotImplementedError
    return model


class PreTrainModel(nn.Module):
    def __init__(self, model_config):
        super(PreTrainModel, self).__init__()
        self._mask_rate = model_config['mask_rate']
        self._encoder_type = model_config['encoder']
        self._decoder_type = model_config['decoder']
        self._drop_edge_rate = model_config['drop_edge_rate']
        self._output_hidden_size = model_config['num_hidden']
        self._concat_hidden = model_config['concat_hidden']
        self._replace_rate = model_config['replace_rate']
        self._mask_token_rate = 1 - self._replace_rate
        encoder_model_config = copy.deepcopy(model_config)
        decoder_model_config = copy.deepcopy(model_config)

        num_hidden = model_config['num_hidden']
        nhead = model_config['num_heads']
        nhead_out = model_config['num_out_heads']
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        enc_num_hidden = num_hidden
        enc_nhead = 1

        # build encoder
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden
        encoder_model_config['m_type'] = self._encoder_type
        encoder_model_config['enc_dec'] = 'encoding'
        encoder_model_config['encoding'] = True
        encoder_model_config['num_hidden'] = enc_num_hidden
        encoder_model_config['out_dim'] = enc_num_hidden
        encoder_model_config['nhead'] = enc_nhead
        encoder_model_config['nhead_out'] = enc_nhead
        self.encoder = setup_model(encoder_model_config)

        # build decoder
        decoder_model_config['m_type'] = self._decoder_type
        decoder_model_config['enc_dec'] = 'decoding'
        decoder_model_config['encoding'] = False
        decoder_model_config['in_dim'] = dec_in_dim
        decoder_model_config['num_hidden'] = dec_num_hidden
        decoder_model_config['out_dim'] = model_config['in_dim']
        decoder_model_config['num_layers'] = 1
        self.decoder = setup_model(decoder_model_config)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, model_config['in_dim']))
        if self._concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * model_config['num_layers'], dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        # setup loss function
        self.criterion = self.setup_loss_fn(model_config['loss_fn'], model_config['alpha_l'])

    def forward(self, g, x):
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)

    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
        enc_rep, all_hidden = self.encoder(x=use_x, edge_index=use_g.edge_index, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0
        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(x=rep, edge_index=pre_use_g.edge_index)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x):
        rep = self.encoder(x=x, edge_index=g.edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
