import torch
import torch.nn as nn


class SupConLoss(nn.Module):

    def __init__(self):
        super(SupConLoss, self).__init__()
        self.device = torch.device("cuda")

    def forward(self, out_1, out_2, batch_size, temperature=0.5):
        # Generate a unit matrix, batch_size of size
        mask = torch.eye(batch_size, dtype=torch.float32)
        mask = mask.repeat(2, 2)

        # Generate a matrix of all 1's with size batch_size * batch_size
        matrix = torch.ones((batch_size * 2, batch_size * 2))

        # Remove diagonals to get masks for negative sample pairs
        logits_mask = matrix.fill_diagonal_(0)
        mask.fill_diagonal_(0)
        logits_mask, mask = logits_mask.to(self.device), mask.to(self.device)
        # Splice the two outputs
        out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]

        # Calculate the similarity matrix and apply temperature scaling
        sim_matrix = torch.mm(out, out.t()) / temperature  # [2*B, 2*B]

        # Negative sample pairs
        exp_logits = torch.exp(sim_matrix) * logits_mask

        log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - 0.5 * mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()

        return loss
