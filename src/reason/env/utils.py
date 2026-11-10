"""Action conversion shared by the environment and search tree."""


def normalize_action(action):
    """Hashable macro action; preserve fractional ages in raw days."""
    return tuple((int(token), float(age)) for token, age in action)


def build_action_groups(action_history):
    return [list(action) for action in action_history]


def flatten_action(action_history):
    return [event for action in action_history for event in action]
