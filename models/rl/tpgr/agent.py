import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from models.modules import FeedforwardNetwork
from util.sampling import sample_pytorch_dist


class TreeAgent(torch.nn.Module):
    def __init__(self, tree, state_size, hidden_dims, output_list_size):
        super().__init__()
        self.tree = tree
        self.k = output_list_size
        self.policies = self._create_policies(state_size, hidden_dims)

    def _create_policies(self, state_size, hidden_dims):
        tree_trajectories = self.tree.trajectory_to_id.keys()
        df = pd.DataFrame(tree_trajectories, columns=["first", "second"])
        num_units = df.groupby("first").count().squeeze().values
        extra_root_policy = [len(num_units)]
        extra_root_policy.extend(num_units)
        all_policies = [FeedforwardNetwork(state_size, hidden_dims, output_dim=n) for n in extra_root_policy]
        return torch.nn.ModuleList(all_policies)

    def forward(self, state, deterministic=False, mask_items=None):
        """
        This is the full policy, could be put in a separate Module (forward pass)
        """
        # Root policy
        logits = self.policies[0](state)
        if not deterministic:
            next_policy_indices, log_probs, _ = sample_pytorch_dist(logits)
            log_probs = log_probs.t()

        else:
            next_policy_indices = torch.topk(logits, 1).indices
            log_probs = logits.log_softmax(dim=-1)
            log_probs = log_probs.gather(1, next_policy_indices)
        next_policy_indices = next_policy_indices.squeeze(0)
        # Policy layer
        output_trajectories = []
        output_log_probs = []
        for e, index in enumerate(next_policy_indices):
            batch_index_state = state[e].unsqueeze(0)
            # The 1 + index here is because this is how we create the tree indices.
            final_logits = self.policies[1 + index.item()](batch_index_state).view(-1)

            probs = final_logits.softmax(dim=-1)
            final_log_probs = final_logits.log_softmax(dim=-1)
            # Mask the zero padding-id which sits at (0,0). Therefore we sample one more item
            # from that particular tree and skip the 0
            # TODO: Full masking of recent items
            if mask_items is not None:
                mask_trajectories = torch.as_tensor(
                    [[self.tree.get_trajectory_for_id(idx.item()) for idx in batch] for batch in mask_items],
                    device=final_logits.device)
                k = mask_trajectories[..., 0] == index
                involved_trajectories = mask_trajectories[k]
                indices = involved_trajectories[..., 1].flatten()
                final_logits[indices] = float("-inf")

            if not deterministic:
                top_k_item_indices = torch.multinomial(probs + 1e-7,
                                                       self.k + 1)  # + 1 so we can mask the 0 if it was drawn
            else:
                top_k_item_indices = torch.topk(final_logits, self.k + 1).indices

            zero_trajectory = self.tree.get_trajectory_for_id(0)
            if index.item() == zero_trajectory[0]:
                top_k_item_indices = top_k_item_indices[top_k_item_indices != zero_trajectory[-1]]
            item_indices = top_k_item_indices[:self.k]
            final_log_probs = final_log_probs.index_select(0, item_indices)

            # Create trajectories and log_probs through stacking with the first decision
            first_index = index.expand(self.k).detach().cpu().numpy()
            trajectories = np.stack([first_index, item_indices.detach().cpu().numpy()], axis=0).T
            final_log_probs = log_probs[e] + final_log_probs.squeeze()
            output_trajectories.append(trajectories)
            output_log_probs.append(final_log_probs)
        output_trajectories = np.stack(output_trajectories)
        output_log_probs = torch.stack(output_log_probs)
        return output_trajectories, output_log_probs

    def log_probs_for_actions(self, state, targets):
        first_layer_logits = self.policies[0](state)
        trajectories = [self.tree.get_trajectory_for_id(idx.item()) for idx in targets]
        first_log_probs = torch.log_softmax(first_layer_logits, dim=-1)
        total_probs = []
        for i, (state, decision) in enumerate(zip(state, trajectories)):
            # first one
            second_layer_logits = self.policies[1 + decision[0]](state)
            second_log_probs = torch.log_softmax(second_layer_logits, dim=-1)

            first_log_prob = first_log_probs[i, decision[0]]
            probs = first_log_prob + second_log_probs[decision[1]]
            if probs.ndimension() == 0:
                probs = probs.unsqueeze(0)
            total_probs.append(probs)
        log_probs = self._stack_or_pad_together(total_probs)
        return targets.unsqueeze(1).detach().cpu().numpy(), log_probs

    @staticmethod
    def _stack_or_pad_together(tensor, padding_value=0):
        lens = [t.shape[-1] for t in tensor]
        all_eqal = all(map(lambda x: lens[0] == x, lens))
        if all_eqal:
            log_probs = torch.stack(tensor)
        else:
            log_probs = pad_sequence(tensor, padding_value=padding_value, batch_first=True)
        return log_probs
