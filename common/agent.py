import copy
import numpy as np
import torch
import torch.nn.functional as F

def default_states_preprocessor(states):
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


class ArgmaxActionSelector:
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class ProbabilityActionSelector:
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class PolicyAgent:
    def __init__(self, model, action_selector=ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def initial_state(self):
        return None

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


class ActorCriticAgent:
    def __init__(self, model, action_selector=ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def initial_state(self):
        return None

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v, values_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        agent_states = values_v.data.squeeze().cpu().numpy().tolist()
        return np.array(actions), agent_states


