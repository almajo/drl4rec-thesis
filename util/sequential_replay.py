import logging
import random
from collections import deque, namedtuple

import numpy as np
import torch

from drlap.utilities.data_structures.Deque import Deque
from drlap.utilities.data_structures.Max_Heap import Max_Heap
from drlap.utilities.data_structures.Replay_Buffer import Replay_Buffer
from util.helpers import unwrap_state

logger = logging.getLogger(__name__)


class Buffer:
    def __init__(self, batch_size, max_len):
        self.deque = deque(maxlen=max_len)
        self.batch_size = batch_size

    def append(self, state_list):
        # filter zeros
        non_zeros = filter(lambda x: 0 not in x[-1], state_list)
        self.deque.extend(non_zeros)

    def sample(self):
        return random.sample(self.deque, k=self.batch_size)

    def __len__(self):
        return len(self.deque)

    def __iter__(self):
        buffer = list(self.deque)
        random.shuffle(buffer)
        for i in range(0, len(buffer), self.batch_size):
            yield buffer[i: i + self.batch_size]


class ReplayBuffer(Replay_Buffer):
    def __init__(self, buffer_size, batch_size, seed, device, **kwargs):
        super().__init__(buffer_size, batch_size, seed, device, **kwargs)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed
        self.device = device

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if isinstance(dones, list) or isinstance(dones, np.ndarray) or isinstance(dones, torch.Tensor):

            experiences = [self.experience(state.detach().cpu().numpy(),
                                           action,
                                           reward.detach().cpu().numpy(),
                                           next_state.detach().cpu().numpy(),
                                           done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        state, action, reward, next_state, done = list(zip(*experiences))
        state = torch.as_tensor(np.stack(state), device=self.device)
        action = torch.as_tensor(action, device=self.device).unsqueeze_(-1)
        reward = torch.as_tensor(np.stack(reward), device=self.device).unsqueeze_(-1)
        next_state = torch.as_tensor(np.stack(next_state), device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze_(-1)
        return state, action, reward, next_state, done

    @property
    def memory_size(self):
        return len(self.memory)


class SequentialReplay(ReplayBuffer):
    def __init__(self, config, seed, device, **kwargs):
        super().__init__(config.hyperparameters.get("Replay")["buffer_size"],
                         config.hyperparameters.get("batch_size"), seed, device)

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list or isinstance(dones, np.ndarray):
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""

        states = [e.state for e in experiences if e is not None]

        states = unwrap_state(states, device=self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)

        next_states = [e.next_state for e in experiences if e is not None]
        next_states = unwrap_state(next_states, device=self.device)

        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences


class SequentialPriorityReplay(Max_Heap, Deque, SequentialReplay):
    """Data structure that maintains a deque, a heap and an array. The deque keeps track of which experiences are the oldest and so
     tells us which ones to delete once the buffer starts getting full. The heap lets us quickly retrieve the experience
     with the max td_value. And the array lets us do quick random samples with probabilities equal to the proportional td errors.
     We also keep track of the sum of the td values using a simple variable.

     Time Complexity:
     - Extracting max td error - O(1)
     - Extracting sum of td errors - O(1)
     - Updating td errors of sample - O(log N)
     - Add experience - O(log N)
     - Sample experiences proportional to TD error - O(1)

     Space Complexity: O(N)
     """

    def __init__(self, config, seed=0, device="cpu", **kwargs):
        logger.info("Initializing PriorityReplay with its Heap")
        buffer_params = config.hyperparameters["Replay"]
        Max_Heap.__init__(self, buffer_params["buffer_size"], dimension_of_value_attribute=5, default_key_to_use=0)
        Deque.__init__(self, buffer_params["buffer_size"], dimension_of_value_attribute=5)
        SequentialReplay.__init__(self, config, seed, device)
        logger.info("Finished initialization")
        self.deques_td_errors = self.initialise_td_errors_array()

        self.heap_index_to_overwrite_next = 1
        self.number_experiences_in_deque = 0
        self.adapted_overall_sum_of_td_errors = 0

        self.alpha = buffer_params["alpha_prioritised_replay"]
        self.beta = buffer_params["beta_prioritised_replay"]
        self.beta_start = self.beta
        self.beta_ascent_factor = (buffer_params["beta_end_value"] - self.beta) / buffer_params["beta_end_at_episode"]
        self.incremental_td_error = buffer_params["incremental_td_error"]
        self.batch_size = config.hyperparameters["batch_size"]

        self.heap_indexes_to_update_td_error_for = None

        self.indexes_in_node_value_tuple = {
            "state": 0,
            "action": 1,
            "reward": 2,
            "next_state": 3,
            "done": 4,
        }

        self.device = device
        self.sample_counter = 0

        if "tb_writer" in kwargs:
            self.tb_writer = kwargs.get("tb_writer")

    def initialise_td_errors_array(self):
        """Initialises a deque of Nodes of length self.max_size"""
        return np.zeros(self.max_size)

    def add_experience(self, state, action, reward, next_state, done):
        """Save an experience in the replay buffer"""
        for s, a, r, n, d in zip(state, action, reward, next_state, done):
            td_error = abs(self.give_max_td_error()) + self.incremental_td_error
            self.update_overall_sum(td_error, self.deque[self.deque_index_to_overwrite_next].key)
            self.update_deque_and_deque_td_errors(td_error, s, a, r, n, d)
            self.update_heap_and_heap_index_to_overwrite()
            self.update_number_experiences_in_deque()
            self.update_deque_index_to_overwrite_next()

    def update_overall_sum(self, new_td_error, old_td_error):
        """Updates the overall sum of td_values present in the buffer"""
        self.adapted_overall_sum_of_td_errors += new_td_error - old_td_error

    def update_deque_and_deque_td_errors(self, td_error, state, action, reward, next_state, done):
        """Updates the deque by overwriting the oldest experience with the experience provided"""
        self.deques_td_errors[self.deque_index_to_overwrite_next] = td_error
        self.add_element_to_deque(td_error, (state, action, reward, next_state, done))

    def add_element_to_deque(self, new_key, new_value):
        """Adds an element to the deque"""
        self.update_deque_node_key_and_value(self.deque_index_to_overwrite_next, new_key, new_value)

    def update_heap_and_heap_index_to_overwrite(self):
        """Updates the heap by rearranging it given the new experience that was just incorporated into it. If we haven't
        reached max capacity then the new experience is added directly into the heap, otherwise a pointer on the heap has
        changed to reflect the new experience so there's no need to add it in"""
        if not self.reached_max_capacity:
            self.update_heap_element(self.heap_index_to_overwrite_next, self.deque[self.deque_index_to_overwrite_next])
            self.deque[self.deque_index_to_overwrite_next].heap_index = self.heap_index_to_overwrite_next
            self.update_heap_index_to_overwrite_next()

        heap_index_change = self.deque[self.deque_index_to_overwrite_next].heap_index
        self.reorganise_heap(heap_index_change)

    def update_heap_index_to_overwrite_next(self):
        """This updates the heap index to write over next. Once the buffer gets full we stop calling this function because
        the nodes the heap points to start being changed directly rather than the pointers on the heap changing"""
        self.heap_index_to_overwrite_next += 1

    def swap_heap_elements(self, index1, index2):
        """Swaps two position of two heap elements and then updates the heap_index stored in the two nodes. We have to override
        this method from Max_Heap so that it also updates the heap_index variables"""
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
        self.heap[index1].heap_index = index1
        self.heap[index2].heap_index = index2

    def sample(self, **kwargs):
        """Randomly samples a batch from experiences giving a higher likelihood to experiences with a higher td error. It then
        calculates an importance sampling weight for each sampled experience, you can read about this in the paper:
        https://arxiv.org/pdf/1511.05952.pdf
        :param **kwargs: """
        # Upgrade sample counter for beta upgrade
        self.sample_counter += 1
        self._update_beta()

        experiences, deque_sample_indexes = self.pick_experiences_based_on_proportional_td_error()
        states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
        self.deque_sample_indexes_to_update_td_error_for = deque_sample_indexes
        importance_sampling_weights = self.calculate_importance_sampling_weights(experiences)
        return states, actions, rewards, next_states, dones, importance_sampling_weights

    def pick_experiences_based_on_proportional_td_error(self):
        """Randomly picks a batch of experiences with probability equal to their proportional td_errors"""
        probabilities = self.deques_td_errors / self.give_adapted_sum_of_td_errors()
        deque_sample_indexes = np.random.default_rng().choice(len(self.deques_td_errors), size=self.batch_size,
                                                              replace=False,
                                                              p=probabilities)
        experiences = self.deque[deque_sample_indexes]
        return experiences, deque_sample_indexes

    def calculate_importance_sampling_weights(self, experiences):
        """Calculates the importance sampling weight of each observation in the sample.
        The weight is proportional to the td_error of the observation,
        see the paper here for more details: https://arxiv.org/pdf/1511.05952.pdf"""
        td_errors = np.array([experience.key for experience in experiences], dtype=np.float32)
        importance_sampling_weights = ((self.number_experiences_in_deque / self.give_adapted_sum_of_td_errors())
                                       * td_errors
                                       ) ** -self.beta
        max_weight = importance_sampling_weights.max()
        importance_sampling_weights /= max_weight
        importance_sampling_weights = torch.from_numpy(importance_sampling_weights).to(self.device)
        return importance_sampling_weights

    def update_td_errors(self, td_errors):
        """Updates the td_errors for the provided heap indexes. The indexes should be the observations provided most
        recently by the give_sample method"""
        for raw_td_error, deque_index in zip(td_errors, self.deque_sample_indexes_to_update_td_error_for):
            td_error = (abs(raw_td_error) + self.incremental_td_error) ** self.alpha
            corresponding_heap_index = self.deque[deque_index].heap_index
            self.update_overall_sum(td_error, self.heap[corresponding_heap_index].key)
            self.heap[corresponding_heap_index].key = td_error
            self.reorganise_heap(corresponding_heap_index)
            self.deques_td_errors[deque_index] = td_error

    def give_max_td_error(self):
        """Returns the maximum td error currently in the heap. Because it is a max heap this is the top element of the heap"""
        return self.give_max_key()

    def give_adapted_sum_of_td_errors(self):
        """Returns the sum of td errors of the experiences currently in the heap"""
        return self.adapted_overall_sum_of_td_errors

    def __len__(self):
        """Tells us how many experiences there are in the replay buffer. This number will never exceed self.max_size"""
        return self.number_experiences_in_deque

    def ready_to_learn(self, batch_size):
        return self.number_experiences_in_deque >= batch_size

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""

        states = [e.value[self.indexes_in_node_value_tuple["state"]] for e in experiences if e is not None]

        states = unwrap_state(states, device=self.device)
        actions = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["action"]] for e in experiences
                                              if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.value[self.indexes_in_node_value_tuple["reward"]] for e in experiences
                                              if e is not None])).float().to(self.device)

        next_states = [e.value[self.indexes_in_node_value_tuple["next_state"]] for e in experiences if e is not None]
        next_states = unwrap_state(next_states, device=self.device)

        dones = torch.from_numpy(np.vstack([int(e.value[self.indexes_in_node_value_tuple["done"]]) for e in experiences
                                            if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def _update_beta(self):
        if self.beta < 1:
            self.beta = self.beta_ascent_factor * self.sample_counter + self.beta_start
