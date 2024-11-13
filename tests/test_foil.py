import os
import sys
import numpy as np
import pytest

HERE  = os.getcwd()
sys.path.append(HERE)

from foil.foil import REFERENCES, VOCABULARY, METHODS, Foil


def generate_parameters(method: str) -> dict:
    if method == 'NACA':
        for relative_camber in np.arange(0, 1, 0.1):
            for x_relative_camber in np.arange(0.1, 1, 0.1):
                for relative_thickness in np.arange(0, 1, 0.1):
                    for closed in (False, True):
                        yield {'relative_camber': relative_camber,
                                'x_relative_camber': x_relative_camber,
                                'relative_thickness': relative_thickness,
                                'closed': closed}
    elif method == 'BMSTU':
        return
    elif method == 'MYNK':
        return


def test_foil_init():
    for method in ('NACA', ):
        for parameters in generate_parameters(method):
            assert Foil(method, **parameters)
            break


def test_foil_naca():
    method = 'NACA'
    for parameters in generate_parameters(method):
        print(parameters)
        assert Foil(method, **parameters).coordinates


def test_foil_show():
    pass
