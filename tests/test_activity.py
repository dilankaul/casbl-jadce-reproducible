import numpy as np
from casbl_jadce.activity_model import event_activation_probabilities


def test_event_probability_is_zero_beyond_D():
    devices = np.array([[0., 0.], [21., 0.]])
    events = np.array([[0., 0.]])
    p = event_activation_probabilities(devices, events, kappa=3.0, D=20.0)
    assert np.isclose(p[0], 1.0)
    assert p[1] == 0.0
