import numpy as np
import pytest

import nengo
import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot, set_color_cycle


@pytest.mark.parametrize("use_eventplot", [True, False])
def test_rasterplot(use_eventplot, Simulator, seed, plt):

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.sin(6 * t))
        a = nengo.Ensemble(100, 1)
        nengo.Connection(u, a)
        ap = nengo.Probe(a.neurons)

    with Simulator(model) as sim:
        sim.run(1.0)

    rasterplot(sim.trange(), sim.data[ap], use_eventplot=use_eventplot)

    # TODO: add assertions


def test_set_color_cycle():
    """Tests if KeyError thrown"""
    set_color_cycle(["red", "green", "blue"], ax=None)
    set_color_cycle(["red", "green", "blue"], ax=plt.gca())


def test_rasterplot_with_empty():
    """Tests rasterplot with an empty T array"""
    with nengo.Network() as net:
        pass

    with nengo.Simulator(net) as sim:
        sim.run(1)

    class Test:
        shape = (1, 0)  # has to be 0
        T = []  # is empty

    fakesim = Test()

    rasterplot(sim.trange(), fakesim, ax=None, use_eventplot=True)
