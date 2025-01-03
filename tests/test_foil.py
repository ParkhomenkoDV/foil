from numpy import pi, linspace, random
import pytest

from foil.foil import REFERENCES, VOCABULARY, METHODS
from foil.foil import Foil
from foil.foil import LinesIntersectionError

NUM = 4
DELTA = 0.000_1


def generate_relative_camber(low=0, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_x_relative_camber(low=0 + DELTA, high=1 - DELTA, num=NUM, endpoint=True):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_relative_thickness(low=0, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_closed():
    for x in (False, True): yield x


def generate_rotation_angle(low=0, high=pi / 2, num=NUM, endpoint=True):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_relative_inlet_radius(low=0, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_relative_outlet_radius(low=0, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_inlet_angle(low=0, high=pi, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_outlet_angle(low=0, high=pi, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_x_ray_cross(low=0 + DELTA, high=1 - DELTA, num=NUM, endpoint=True):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_upper_proximity(low=0, high=1, num=NUM, endpoint=True):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_mynk_coefficient(low=0, high=2, num=NUM, endpoint=True):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_x_relative_camber_upper(low=0 + DELTA, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_x_relative_camber_lower(low=0 + DELTA, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_relative_camber_upper(low=-1 + DELTA, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_relative_camber_lower(low=-1 + DELTA, high=1, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_d2y_dx2_upper():
    for _ in range(NUM): yield random.randint(-10, 10)


def generate_d2y_dx2_lower():
    for _ in range(NUM): yield random.randint(-10, 10)


def generate_theta_outlet_upper(low=-pi / 2 + DELTA, high=pi / 2, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_theta_outlet_lower(low=-pi / 2 + DELTA, high=pi / 2, num=NUM, endpoint=False):
    for x in linspace(low, high, num, endpoint=endpoint): yield x


def generate_points(low=3, high=1_00):
    for n in range(low, high): yield random.random((n, 2))


def generate_deg(low=1, high=3, endpoint=True):
    for deg in range(low, high + 1 if endpoint else high): yield deg


def generate_parameters(method: str):
    if method == 'NACA':
        for relative_camber in generate_relative_camber():
            for x_relative_camber in generate_x_relative_camber():
                for relative_thickness in generate_relative_thickness():
                    for closed in generate_closed():
                        yield {'relative_camber': relative_camber,
                               'x_relative_camber': x_relative_camber,
                               'relative_thickness': relative_thickness,
                               'closed': closed}
    elif method == 'BMSTU':
        for rotation_angle in generate_rotation_angle():
            for relative_inlet_radius in generate_relative_inlet_radius():
                for relative_outlet_radius in generate_relative_outlet_radius():
                    for inlet_angle in generate_inlet_angle():
                        for outlet_angle in generate_outlet_angle():
                            for x_ray_cross in generate_x_ray_cross():
                                for upper_proximity in generate_upper_proximity():
                                    yield {'rotation_angle': rotation_angle,
                                           'relative_inlet_radius': relative_inlet_radius,
                                           'relative_outlet_radius': relative_outlet_radius,
                                           'inlet_angle': inlet_angle,
                                           'outlet_angle': outlet_angle,
                                           'x_ray_cross': x_ray_cross,
                                           'upper_proximity': upper_proximity}
    elif method == 'MYNK':
        for mynk_coefficient in generate_mynk_coefficient():
            yield {'mynk_coefficient': mynk_coefficient}
    elif method == 'PARSEC':
        for relative_inlet_radius in generate_relative_inlet_radius():
            for x_relative_camber_upper in generate_x_relative_camber_upper():
                for x_relative_camber_lower in generate_x_relative_camber_lower():
                    for relative_camber_upper in generate_relative_camber_upper():
                        for relative_camber_lower in generate_relative_camber_lower():
                            for d2y_dx2_upper in generate_d2y_dx2_upper():
                                for d2y_dx2_lower in generate_d2y_dx2_lower():
                                    for theta_outlet_upper in generate_theta_outlet_upper():
                                        for theta_outlet_lower in generate_theta_outlet_lower():
                                            yield {'relative_inlet_radius': relative_inlet_radius,
                                                   'x_relative_camber_upper': x_relative_camber_upper,
                                                   'x_relative_camber_lower': x_relative_camber_lower,
                                                   'relative_camber_upper': relative_camber_upper,
                                                   'relative_camber_lower': relative_camber_lower,
                                                   'd2y_dx2_upper': d2y_dx2_upper,
                                                   'd2y_dx2_lower': d2y_dx2_lower,
                                                   'theta_outlet_upper': theta_outlet_upper,
                                                   'theta_outlet_lower': theta_outlet_lower}
    elif method == 'BEZIER':
        for points in generate_points():
            yield {'points': points}
    elif method == 'MANUAL':
        for deg in generate_deg():
            for points in generate_points(4):
                yield {'points': points, 'deg': deg}


def test_foil_init():
    for method in ('NACA', 'BMSTU', 'MYNK', 'PARSEC', 'BEZIER',):
        for parameters in generate_parameters(method):
            assert Foil(method, **parameters)
            break


def test_foil_naca():
    method = 'NACA'
    for parameters in generate_parameters(method):
        assert Foil(method, **parameters).coordinates


def test_foil_bmstu():
    method = 'BMSTU'
    for parameters in generate_parameters(method):
        try:
            assert Foil(method, **parameters).coordinates
        except LinesIntersectionError:
            pass  # pytest.skip(LinesIntersectionError.__name__)  # пропскается все оставшиеся тесты


def test_foil_mynk():
    method = 'MYNK'
    for parameters in generate_parameters(method):
        assert Foil(method, **parameters).coordinates


def test_foil_parsec():
    method = 'PARSEC'
    for parameters in generate_parameters(method):
        assert Foil(method, **parameters).coordinates


def test_foil_bezier():
    method = 'BEZIER'
    for parameters in generate_parameters(method):
        assert Foil(method, **parameters).coordinates


def test_foil_manual():
    method = 'MANUAL'
    for parameters in generate_parameters(method):
        assert Foil(method, **parameters).coordinates


def test_foil_circle_foil():
    method = 'CIRCLE'


def test_foil_circle_channel():
    method = 'CIRCLE'


def test_foil_circle():
    method = 'CIRCLE'


def test_foil_show():
    pass
