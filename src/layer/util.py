from torch import nn


def get_activation(activation):
    """
    Get the activation function from a string.
    If anything else is passed, just return it.
     (intention: a callable such as torch.nn.ReLU() would be returned unchanged)
    """
    if isinstance(activation, str):
        try:
            activation = getattr(
                nn, activation
            )()  # todo: should this really have parentheses?
        except AttributeError:
            raise ValueError(f"The activation '{activation}' does not exist in pytorch")
    return activation
