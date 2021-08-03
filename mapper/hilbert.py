import numpy as np

# dictionary containing the first order hilbert curves
base_shape = {'u': [np.array([0, 1]), np.array([1, 0]), np.array([0, -1])],
              'd': [np.array([0, -1]), np.array([-1, 0]), np.array([0, 1])],
              'r': [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])],
              'l': [np.array([-1, 0]), np.array([0, -1]), np.array([1, 0])]}


def hilbert_curve(order, orientation):
    """
    Recursively creates the structure for a hilbert curve of given order
    """
    if order > 1:
        if orientation == 'u':
            return hilbert_curve(order - 1, 'r') + [np.array([0, 1])] + \
                   hilbert_curve(order - 1, 'u') + [np.array([1, 0])] + \
                   hilbert_curve(order - 1, 'u') + [np.array([0, -1])] + \
                   hilbert_curve(order - 1, 'l')
        elif orientation == 'd':
            return hilbert_curve(order - 1, 'l') + [np.array([0, -1])] + \
                   hilbert_curve(order - 1, 'd') + [np.array([-1, 0])] + \
                   hilbert_curve(order - 1, 'd') + [np.array([0, 1])] + \
                   hilbert_curve(order - 1, 'r')
        elif orientation == 'r':
            return hilbert_curve(order - 1, 'u') + [np.array([1, 0])] + \
                   hilbert_curve(order - 1, 'r') + [np.array([0, 1])] + \
                   hilbert_curve(order - 1, 'r') + [np.array([-1, 0])] + \
                   hilbert_curve(order - 1, 'd')
        else:
            return hilbert_curve(order - 1, 'd') + [np.array([-1, 0])] + \
                   hilbert_curve(order - 1, 'l') + [np.array([0, -1])] + \
                   hilbert_curve(order - 1, 'l') + [np.array([1, 0])] + \
                   hilbert_curve(order - 1, 'u')
    else:
        return base_shape[orientation]

def hilbert_map(order):
    curve = hilbert_curve(order, 'u')

    curve = np.array(curve)
    cumulative_curve = np.array([np.sum(curve[:i], 0) for i in range(len(curve)+1)])
    return cumulative_curve