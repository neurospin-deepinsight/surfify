import torch
from torch import nn
import torch.nn.functional as func


class SimCLR(nn.Module):
    """ Class implementing the simCLR model.
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations
    """

    def __init__(self, latent_dim, hidden_layers, temperature, backbone,
                 return_logits=False):
        super().__init__()

        self.backbone = backbone
        # projector
        sizes = [latent_dim] + hidden_layers
        layers = []
        for i in range(len(sizes) - 2):
            layers.extend([
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, y1, y2):
        z_i = self.projector(self.backbone(y1))
        z_j = self.projector(self.backbone(y2))
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]
        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature
        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature
        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)
