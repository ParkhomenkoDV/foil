import numpy as np
import pytest

from foil.foil import REFERENCES, VOCABULARY, METHODS, Foil


STEP = 0.1
NUM = 10

def generate_relative_camber(low=0, high=1, step=STEP):
    for x in np.arange(low, high, step): yield x

def generate_x_relative_camber(low=0.01, high=0.99, step=STEP):
    for x in np.linspace(low, high, NUM): yield x


def generate_parameters(method: str) -> dict:
    if method == 'NACA':
        for relative_camber in generate_relative_camber():
            for x_relative_camber in generate_x_relative_camber():
                for relative_thickness in np.arange(0, 1, 0.1):
                    for closed in (False, True):
                        yield {'relative_camber': relative_camber,
                                'x_relative_camber': x_relative_camber,
                                'relative_thickness': relative_thickness,
                                'closed': closed}
        '''
    elif method == 'BMSTU':
        for rotation_angle in :
        for relative_inlet_radius in :
        for relative_outlet_radius in :
        for inlet_angle in :
        for outlet_angle in :
        for  x_ray_cross in :
        for upper_proximity in :
    '''
    elif method == 'MYNK':
        return


def test_foil_init():
    for method in ('NACA', ): # METHODS
        for parameters in generate_parameters(method):
            assert Foil(method, **parameters)
            break


def test_foil_naca():
    method = 'NACA'
    for parameters in generate_parameters(method):
        assert Foil(method, **parameters).coordinates


def test_foil_show():
    pass