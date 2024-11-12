import os
import sys
import time
from types import MappingProxyType  # неизменяемый словарь
import warnings

from tqdm import tqdm
from colorama import Fore
import pandas as pd
import numpy as np
from numpy import array, arange, linspace, zeros, full, zeros_like, full_like
from numpy import nan, isnan, inf, isinf, pi
from numpy import cos, sin, tan, arctan as atan, sqrt, floor, ceil, radians, degrees
from scipy import interpolate, integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from decorators import timeit, warns
from mathematics import derivative, Axis
from mathematics import coordinate_intersection_lines, coefficients_line, angle_between, distance, distance2line
from mathematics import cot, tan2cos, tan2sin

HERE = os.path.dirname(__file__)
sys.path.append(HERE)

from curves import bernstein_curve

# Список использованной литературы
REFERENCES = MappingProxyType({
    1: '''Теория и проектирование газовой турбины: учебное пособие /
В.Е. Михальцев, В.Д. Моляков; под ред. А.Ю. Вараксина. -
Москва: Издательство МГТУ им. Н.Э. Баумана, 2020. - 230, [2] с.: ил.''',
})

# словарь терминов их описания, единицы измерения и граничные значения
VOCABULARY = MappingProxyType({
    'method': {
        'description': 'название профиля',
        'unit': '[]',
        'type': (str,),
        'assert': (lambda method:
                   '' if method.strip().upper() in ("NACA", "BMSTU", "MYNK", "PARSEC", "BEZIER", "MANUAL", "CIRCLE")
                   else 'method not in ("NACA", "BMSTU", "MYNK", "PARSEC", "BEZIER", "MANUAL", "CIRCLE")',), },
    'discreteness': {
        'description': 'дискретность',
        'unit': '[]',
        'type': (int, np.integer),
        'assert': (lambda discreteness: '' if 3 <= discreteness else '3 <= discreteness',), },
    'relative_step': {
        'description': 'относительный шаг решетки профилей',
        'unit': '[]',
        'type': (float, int, np.number),
        'assert': (lambda relative_step: '' if 0 < relative_step else '0 < relative_step',), },
    'installation_angle': {
        'description': 'угол установки профиля',
        'unit': '[]',
        'type': (float, int, np.number),
        'assert': (lambda installation_angle:
                   '' if -pi / 2 <= installation_angle <= pi / 2 else '-pi / 2 <= installation_angle <= pi / 2',), },
    'name': {
        'description': 'название профиля',
        'unit': '[]',
        'type': (str,),
        'assert': (lambda name: '' if all(symbol not in name for symbol in ".$#*{}\|/?:><%")
        else 'all(symbol not in name for symbol in ".$#*{}\|/?:><%"',), },
    'parameters': {
        'description': 'параметры профиля',
        'unit': '[]',
        'type': (dict,),
        'assert': (lambda parameters: '' if len(parameters) > 0 else 'parameters is empty',), },
    'coordinates': {
        'description': 'координаты профиля считая от выходной кромки против часовой стрелки',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda coordinates: '' if 3 <= len(coordinates) else '3 <= len(coordinates)',
                   lambda coordinates:
                   '' if all(isinstance(coordinate, (tuple, list, np.ndarray)) for coordinate in coordinates)
                   else 'all(isinstance(coordinate, (tuple, list, np.ndarray)) for coordinate in coordinates)',
                   lambda coordinates: '' if all(len(coordinate) == 2 for coordinate in coordinates)
                   else 'all(len(coordinate) == 2 for coordinate in coordinates)',
                   lambda coordinates:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in coordinates)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))'
                        'for x, y in coordinates)',
                   lambda coordinates:
                   '' if tuple(coordinates[:np.argmin(array(coordinates).T[0])]) ==
                         tuple(sorted(coordinates[:np.argmin(array(coordinates).T[0])],
                                      key=lambda point: point[0], reverse=True)) and
                         tuple(coordinates[np.argmin(array(coordinates).T[0]):]) ==
                         tuple(sorted(coordinates[np.argmin(array(coordinates).T[0]):],
                                      key=lambda point: point[0], reverse=False))
                   else 'ascending error',), },
    'upper': {
        'description': 'координаты спинки',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda upper: '' if 3 <= len(upper) else '3 <= len(upper)',
                   lambda upper: '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in upper)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in upper)',
                   lambda upper: '' if all(len(coord) == 2 for coord in upper)
                   else 'all(len(coord) == 2 for coord in upper)',
                   lambda upper:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in upper)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))'
                        'for x, y in upper)',
                   lambda upper:
                   '' if tuple(upper) == tuple(sorted(upper, key=lambda point: point[0], reverse=False))
                   else 'tuple(upper) == tuple(sorted(upper, key=lambda point: point[0], reverse=False))',), },
    'lower': {
        'description': 'координаты корыта',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda lower: '' if 3 <= len(lower) else '3 <= len(lower)',
                   lambda lower: '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in lower)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in lower)',
                   lambda lower: '' if all(len(coord) == 2 for coord in lower)
                   else 'all(len(coord) == 2 for coord in lower)',
                   lambda lower:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in lower)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))'
                        'for x, y in lower)',
                   lambda lower:
                   '' if tuple(lower) == tuple(sorted(lower, key=lambda point: point[0], reverse=False))
                   else 'tuple(lower) == tuple(sorted(lower, key=lambda point: point[0], reverse=False))',), },

    'rotation_angle': {
        'description': 'угол поворота потока',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda rotation_angle: '' if 0 < rotation_angle <= pi else f'0 < rotation_angle <= {pi}',), },
    'relative_inlet_radius': {
        'description': 'относительный радиус входной кромки',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda relative_inlet_radius:
                   '' if 0 <= relative_inlet_radius < 1 else '0 <= relative_inlet_radius < 1',), },
    'relative_outlet_radius': {
        'description': 'относительный радиус выходной кромки',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda relative_outlet_radius:
                   '' if 0 <= relative_outlet_radius < 1 else '0 <= relative_outlet_radius < 1',), },
    'closed': {
        'description': 'замкнутость профиля',
        'unit': '[]',
        'type': (bool,),
        'assert': tuple(), },
    'inlet_angle': {
        'description': 'угол раскрытия входной кромки',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda inlet_angle: '' if 0 <= inlet_angle < pi else f'0 <= inlet_angle < {pi}',), },
    'outlet_angle': {
        'description': 'угол раскрытия выходной кромки',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda outlet_angle: '' if 0 <= outlet_angle < pi else f'0 <= outlet_angle < {pi}',), },
    'x_ray_cross': {
        'description': 'относительная координата х пересечения входного и выходного лучей',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_ray_cross: '' if 0 < x_ray_cross < 1 else '0 < x_ray_cross < 1',), },
    'upper_proximity': {
        'description': 'степень приближенности к спинке',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda upper_proximity: '' if 0 <= upper_proximity <= 1 else '0 <= upper_proximity <= 1',), },
    'relative_thickness': {
        'description': 'максимальная относительная толщина',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_thickness: '' if 0 <= relative_thickness < 1 else '0 <= relative_thickness < 1',), },
    'x_relative_camber': {
        'description': 'относительна координата х максимальной выпуклости',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_relative_camber: '' if 0 < x_relative_camber < 1 else '0 < x_relative_camber < 1',), },
    'relative_camber': {
        'description': 'относительная максимальная выпуклость',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_camber: '' if 0 <= relative_camber < 1 else '0 <= relative_camber < 1',), },
    'mynk_coefficient': {
        'description': 'коэффициент Мунка',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda mynk_coefficient: '' if 0 <= mynk_coefficient else '0 <= mynk_coefficient',), },
    'x_relative_camber_upper': {
        'description': 'относительна координата х максимальной выпуклости спинки',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_relative_camber_upper:
                   '' if 0 < x_relative_camber_upper < 1 else '0 < x_relative_camber_upper < 1',), },
    'x_relative_camber_lower': {
        'description': 'относительна координата х максимальной выпуклости корыта',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_relative_camber_lower:
                   '' if 0 < x_relative_camber_lower < 1 else '0 < x_relative_camber_lower < 1',), },
    'relative_camber_upper': {
        'description': 'максимальная относительная толщина спинки относительно оси х',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_camber_upper:
                   '' if -1 < relative_camber_upper < 1 else '-1 < relative_camber_upper < 1',), },
    'relative_camber_lower': {
        'description': 'максимальная относительная толщина корыта относительно оси х',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_camber_lower:
                   '' if -1 < relative_camber_lower < 1 else '-1 < relative_camber_lower < 1',), },
    'd2y_dx2_upper': {
        'description': 'кривизна спинки (вторая производная поверхности)',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': tuple(), },
    'd2y_dx2_lower': {
        'description': 'кривизна корыта (вторая производная поверхности)',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': tuple(), },
    'theta_outlet_upper': {
        'description': 'угол выхода между поверхностью спинки и горизонталью',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda theta_outlet_upper:
                   '' if -pi / 2 < theta_outlet_upper < pi / 2 else f'{-pi / 2} < theta_outlet_upper < {pi / 2}',), },
    'theta_outlet_lower': {
        'description': 'угол выхода между поверхностью корыта и горизонталью',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda theta_outlet_lower:
                   '' if -pi / 2 < theta_outlet_lower < pi / 2 else f'{-pi / 2} < theta_outlet_lower < {pi / 2}',), },
    'points': {
        'description': 'координаты точек или полюсов',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda points: '' if 3 <= len(points) else '3 <= len(points)',
                   lambda points: '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in points)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in points)',
                   lambda points: '' if all(len(coord) == 2 for coord in points)
                   else 'all(len(coord) == 2 for coord in points)',
                   lambda points:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in points)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)) '
                        'for x, y in points)',
                   lambda points:
                   '' if all(points[i][0] > points[i + 1][0] for i in range(0, np.argmin(array(points).T[0]))) and
                         all(points[i][0] < points[i + 1][0] for i in
                             range(np.argmin(array(points).T[0]), len(points) - 1))
                   else 'duplicates ascending error',), },
    'deg': {
        'description': 'степень интерполяции полинома',
        'unit': '[]',
        'type': (int, np.integer),
        'assert': (lambda deg: '' if 1 <= deg <= 3 else '1 <= deg <= 3',), },
    'relative_circles': {
        'description': 'относительные окружности профиля или канала',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda relative_circles: '' if 3 <= len(relative_circles) else '3 <= len(relative_circles)',
                   lambda relative_circles:
                   '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in relative_circles)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in relative_circles)',
                   lambda relative_circles: '' if all(len(coord) == 2 for coord in relative_circles)
                   else 'all(len(coord) == 2 for coord in relative_circles)',
                   lambda relative_circles:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in relative_circles)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)) '
                        'for x, y in relative_circles)',
                   lambda relative_circles: '' if all(0 < x < 1 and 0 < y for x, y in relative_circles)
                   else 'all(0 < x < 1 and 0 < y for x, y in relative_circles)',), },
    'is_airfoil': {
        'description': 'указатель на профиль',
        'unit': '[]',
        'type': (bool,),
        'assert': tuple(), },
})


class Foil:
    """Относительный профиль"""

    __RND = 4  # количество значащих цифр
    __METHODS = {
        'NACA': {'description': '''Четырёхзначные профиль крыла NACA определяется следующим образом:
Первая цифра обозначает максимальный прогиб в процентах от хорды.
Вторая цифра, описывающая расстояние максимального изгиба от передней кромки аэродинамического профиля в десятых долях хорды.
Последние две цифры обозначают максимальную толщину аэродинамического профиля в процентах от хорды.''',
                 'attributes': {
                     'x_relative_camber': VOCABULARY['x_relative_camber'],
                     'relative_camber': VOCABULARY['relative_camber'],
                     'relative_thickness': VOCABULARY['relative_thickness'],
                     'closed': VOCABULARY['closed'], }},
        'BMSTU': {
            'description': f'[{REFERENCES[1]}, с.110-115]',
            'attributes': {
                'rotation_angle': VOCABULARY['rotation_angle'],
                'relative_inlet_radius': VOCABULARY['relative_inlet_radius'],
                'relative_outlet_radius': VOCABULARY['relative_outlet_radius'],
                'inlet_angle': VOCABULARY['inlet_angle'],
                'outlet_angle': VOCABULARY['outlet_angle'],
                'x_ray_cross': VOCABULARY['x_ray_cross'],
                'upper_proximity': VOCABULARY['upper_proximity']}},
        'MYNK': {'description': 'профиль Мунка',
                 'attributes': {
                     'mynk_coefficient': VOCABULARY['mynk_coefficient'], }},
        'PARSEC': {'description': '',
                   'attributes': {
                       'relative_inlet_radius': VOCABULARY['relative_inlet_radius'],
                       'x_relative_camber_upper': VOCABULARY['x_relative_camber_upper'],
                       'x_relative_camber_lower': VOCABULARY['x_relative_camber_lower'],
                       'relative_camber_upper': VOCABULARY['relative_camber_upper'],
                       'relative_camber_lower': VOCABULARY['relative_camber_lower'],
                       'd2y_dx2_upper': VOCABULARY['d2y_dx2_upper'],
                       'd2y_dx2_lower': VOCABULARY['d2y_dx2_lower'],
                       'theta_outlet_upper': VOCABULARY['theta_outlet_upper'],
                       'theta_outlet_lower': VOCABULARY['theta_outlet_lower'], }},
        'BEZIER': {'description': 'профиль, построенный по кривой Безье',
                   'aliases': ('BEZIER', 'БЕЗЬЕ'),
                   'attributes': {
                       'points': VOCABULARY['points'], }},
        'MANUAL': {'description': 'профиль, образованный интерполяцией точек спинки и корыта',
                   'attributes': {
                       'points': VOCABULARY['points'],
                       'deg': VOCABULARY['deg'], }},
        'CIRCLE': {'description': 'профиль, образованный интерполяцией точек канала профильной решетки',
                   'attributes': {
                       'relative_circles': VOCABULARY['relative_circles'],
                       'rotation_angle': VOCABULARY['rotation_angle'],
                       'x_ray_cross': VOCABULARY['x_ray_cross'],
                       'is_airfoil': VOCABULARY['is_airfoil'], }}, }
    __DISCRETENESS = 30  # рекомендуемое количество дискретных точек
    __RELATIVE_STEP = 1.0  # дефолтный относительный шаг []
    __INSTALLATION_ANGLE = 0.0  # дефолтный угол установки [рад]

    __slots__ = ('__method',  # необходимый параметр
                 '__discreteness',
                 '__relative_step', '__installation_angle',
                 '__name', '__parameters',
                 '__coordinates0', '__coordinates', '__x', '__y', '__chord', '__properties', '__channel')

    @property
    def methods(self):
        return Foil.__METHODS


    @property
    def rnd(self) -> int:
        return Foil.__RND

    @rnd.setter
    def rnd(self, value) -> None:
        assert isinstance(value, int) and 0 <= value
        Foil.__rnd = value

    @classmethod
    def validate(cls, **kwargs) -> None:
        """Проверка верности ввода атрибутов профиля"""

        for key, value in kwargs.items():
            assert isinstance(value, VOCABULARY[key]['type'])
            for ass in VOCABULARY[key]['assert']: assert not ass(value), ass(value)

        method, parameters = kwargs.pop('method', None), kwargs.pop('parameters', None)
        if method is not None and parameters is not None:
            for attr in cls.__METHODS[method]['attributes']:
                assert attr in parameters, f'{attr} not in parameters'
                value = parameters.get(attr, None)
                assert isinstance(value, cls.__METHODS[method]['attributes'][attr]['type']), \
                    f'type({attr}) not in {cls.__METHODS[method]["attributes"][attr]["type"]}'
                for ass in cls.__METHODS[method]['attributes'][attr]["assert"]: assert not ass(value), ass(value)

    def __init__(self, method: str, discreteness: int = __DISCRETENESS,
                 relative_step: float | int = __RELATIVE_STEP, installation_angle: float | int = __INSTALLATION_ANGLE,
                 name: str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                 **parameters):
        self.validate(method=method, discreteness=discreteness,
                      relative_step=relative_step, installation_angle=installation_angle,
                      name=name,
                      parameters=parameters)

        self.__method = method.strip().upper()  # метод построения аэродинамического профиля
        self.__discreteness = int(discreteness)  # количество точек дискретизации

        self.__relative_step = float(relative_step)  # относительный шаг []
        self.__installation_angle = float(installation_angle)  # угол установки [рад]

        self.__name = name.strip()  # название профиля
        self.__parameters = parameters  # параметры профиля

        self.reset()

    def __str__(self) -> str:
        return self.__name

    def __setattr__(self, key, value):
        """При установке новых атрибутов расчет обнуляется"""
        if not key.startswith('_'): self.reset()
        object.__setattr__(self, key, value)

    def reset(self):
        """Сброс расчетов"""
        self.__coordinates = tuple()  # относительные координаты профиля считая против часовой стрелки с выходной кромки
        self.__x, self.__y = tuple(), tuple()  # относительные координты x и y профиля
        self.__chord = None  # длина хорды
        self.__properties = dict()  # относительные характеристики профиля
        self.__channel = tuple()  # дифузорность/конфузорность решетки

    @property
    def method(self) -> str:
        return self.__method

    @property
    def discreteness(self) -> int:
        return self.__discreteness

    @discreteness.setter
    def discreteness(self, value: int) -> None:
        self.validate(discreteness=value)
        self.__discreteness = value
        self.reset()

    @property
    def relative_step(self) -> float | int | np.number:
        return self.__relative_step

    @relative_step.setter
    def relative_step(self, value):
        self.validate(relative_step=value)
        self.__relative_step = value
        self.__properties = dict()  # относительные характеристики профиля
        self.__channel = dict()  # дифузорность/конфузорность решетки

    @relative_step.deleter
    def relative_step(self):
        self.__relative_step = Foil.__RELATIVE_STEP
        self.__properties = dict()  # относительные характеристики профиля
        self.__channel = dict()  # дифузорность/конфузорность решетки

    @property
    def installation_angle(self) -> float | int | np.number:
        return self.__installation_angle

    @installation_angle.setter
    def installation_angle(self, value):
        self.validate(installation_angle=value)
        self.__installation_angle = value
        self.reset()

    @installation_angle.deleter
    def installation_angle(self):
        self.__installation_angle = Foil.__INSTALLATION_ANGLE
        self.reset()

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        self.validate(name=value)
        self.__name = value

    @name.deleter
    def name(self) -> None:
        self.__name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    @property
    def parameters(self) -> dict:
        return self.__parameters

    @parameters.setter
    def parameters(self, value: dict):
        self.validate(method=self.method, parameters=value)
        self.__parameters = value
        self.reset()

    @property
    def coordinates(self) -> tuple[tuple[float, float], ...]:
        """Координаты аэродинамического профиля считая от выходной кромки против часовой стрелки"""
        if len(self.__coordinates) == 0: self.__fit()
        return self.__coordinates

    def xy(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Координаты x и y аэродинамического профиля считая от выходной кромки против часовой стрелки"""
        if 0 < len(self.__x) and 0 < len(self.__y): return self.__x, self.__y
        X, Y = array(self.coordinates, dtype='float64').T
        self.__x, self.__y = tuple([float(x) for x in X]), tuple([float(y) for y in Y])
        return self.__x, self.__y

    def function_upper(self, kind, fill_value='extrapolate') -> interpolate.interp1d:
        """Функция спинки аэродинамического профиля"""
        assert isinstance(kind, int) and 1 <= kind <= 3
        upper = self.upper_lower(self.coordinates)['upper']
        return interpolate.interp1d(*array(upper, dtype='float64').T, kind=kind, fill_value=fill_value)

    def function_lower(self, kind, fill_value='extrapolate') -> interpolate.interp1d:
        """Функция корыта аэродинамического профиля"""
        assert isinstance(kind, int) and 1 <= kind <= 3
        lower = self.upper_lower(self.coordinates)['lower']
        return interpolate.interp1d(*array(lower, dtype='float64').T, kind=kind, fill_value=fill_value)

    @classmethod
    def load(cls, points, deg: int, discreteness: int = __DISCRETENESS,
             name: str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) -> object:
        """Загрузка координат профиля"""
        cls.validate(method='MANUAL', discreteness=discreteness, name=name,
                     parameters={'points': points, 'deg': deg})
        return Foil('MANUAL', discreteness, name=name, points=points, deg=deg)

    @classmethod
    def read(cls, path: str, header: bool = True) -> object:
        """Считывание координат профиля из файла"""
        assert os.path.isfile(path), f'file "{path}" does not exist'
        name, extension = os.path.splitext(path)
        extension = extension.lower()

        if extension == 'pkl':
            points = pd.read_pickle(path)
        elif extension in ('txt', 'csv'):
            points = pd.read_csv(path, header=header)
        elif extension == 'xlsx':
            points = pd.read_excel(path, header=header)
        else:
            raise Exception(f'extension "{extension}" is not found')

        return Foil('MANUAL', name=name, points=points)

    def to_dataframe(self) -> pd.DataFrame:
        """Перевод координат в pandas.DataFrame"""
        return pd.DataFrame(self.coordinates, columns=('x', 'y'))

    def write(self, extension: str, index: bool = False, header: bool = True) -> None:
        """Экспортирование координат и характеристик профиля"""
        assert isinstance(extension, str)
        extension = extension.strip().lower().replace('.', '')

        if not os.path.isdir('datas'): os.mkdir('datas')
        ctime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        coordinates, properties = self.to_dataframe(), pd.DataFrame(self.properties, index=[0])

        if extension == 'pkl':
            coordinates.to_pickle(f'datas/foil_coordinates_{ctime}.{extension}', index=index, header=header)
            properties.to_pickle(f'datas/foil_properties_{ctime}.{extension}', index=index, header=header)
        elif extension in ('csv', 'txt'):
            coordinates.to_csv(f'datas/foil_coordinates_{ctime}.{extension}', index=index, header=header)
            properties.to_csv(f'datas/foil_properties_{ctime}.{extension}', index=index, header=header)
        elif extension == 'xlsx':
            coordinates.to_excel(f'datas/foil_coordinates_{ctime}.{extension}', index=index, header=header)
            properties.to_excel(f'datas/foil_properties_{ctime}.{extension}', index=index, header=header)
        else:
            raise Exception(f'extension "{extension}" is not found')

    def __naca(self, discreteness: int,
               relative_thickness, x_relative_camber, relative_camber, closed) -> tuple[tuple[float, float], ...]:

        i = arange(discreteness)  # массив индексов
        betta = i * pi / (2 * (discreteness - 1))
        x = 1 - cos(betta)

        mask = (0 <= x) & (x <= x_relative_camber)

        yf = full_like(i, relative_camber, dtype='float64')
        yf[mask] *= x_relative_camber ** (-2) * (2 * x_relative_camber * x[mask] - x[mask] ** 2)
        yf[~mask] *= ((1 - x_relative_camber) ** (-2) *
                      (1 - 2 * x_relative_camber + 2 * x_relative_camber * x[~mask] - x[~mask] ** 2))

        gradYf = 2 * relative_camber

        coefs = array((0.2969, -0.126, -0.3516, 0.2843, -0.1036 if closed else -0.1015), dtype='float64')

        yc = relative_thickness / 0.2 * np.dot(coefs, np.column_stack((sqrt(x), x, x ** 2, x ** 3, x ** 4)).T)

        tetta = atan(gradYf)

        sin_tetta, cos_tetta = sin(tetta), cos(tetta)  # предварительный расчет для ускорения работы

        X = np.hstack(((x - yc * sin_tetta)[::-1], (x + yc * sin_tetta)[1::]))  # revers против часовой стрелки
        Y = np.hstack(((yf + yc * cos_tetta)[::-1], (yf - yc * cos_tetta)[1::]))  # удаление дубликатной входной точки

        scale = abs(X.max() - X.min())

        return self.transform(tuple(((x, y) for x, y in zip(X, Y))), x0=X.min(), scale=(1 / scale))

    def __bmstu(self, discreteness: int,
                rotation_angle,
                relative_inlet_radius, relative_outlet_radius,
                inlet_angle, outlet_angle, x_ray_cross,
                upper_proximity) -> tuple[tuple[float, float], ...]:

        airfoil_rotation_angle = pi - rotation_angle  # угол поворота профиля

        # tan угла входа и выхода потока
        k_inlet = 1 / (2 * x_ray_cross / (x_ray_cross - 1) * tan(airfoil_rotation_angle))
        k_outlet = 1 / (2 * tan(airfoil_rotation_angle))
        if tan(airfoil_rotation_angle) * airfoil_rotation_angle > 0:
            k_inlet *= ((x_ray_cross / (x_ray_cross - 1) - 1) -
                        sqrt((x_ray_cross / (x_ray_cross - 1) - 1) ** 2 -
                             4 * (x_ray_cross / (x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))
            k_outlet *= ((x_ray_cross / (x_ray_cross - 1) - 1) -
                         sqrt((x_ray_cross / (x_ray_cross - 1) - 1) ** 2 -
                              4 * (x_ray_cross / (x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))
        else:
            k_inlet *= ((x_ray_cross / (x_ray_cross - 1) - 1) +
                        sqrt((x_ray_cross / (x_ray_cross - 1) - 1) ** 2 -
                             4 * (x_ray_cross / (x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))
            k_outlet *= ((x_ray_cross / (x_ray_cross - 1) - 1) +
                         sqrt((x_ray_cross / (x_ray_cross - 1) - 1) ** 2 -
                              4 * (x_ray_cross / (x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))

        # углы входа и выхода профиля
        if airfoil_rotation_angle > 0:
            g_u_inlet, g_d_inlet = ((1 - upper_proximity) * inlet_angle, upper_proximity * inlet_angle)
            g_u_outlet, g_d_outlet = ((1 - upper_proximity) * outlet_angle, upper_proximity * outlet_angle)
        else:
            g_u_inlet, g_d_inlet = upper_proximity * inlet_angle, (1 - upper_proximity) * inlet_angle,
            g_u_outlet, g_d_outlet = upper_proximity * outlet_angle, (1 - upper_proximity) * outlet_angle

        # положения центров окружностей входной и выходной кромок
        O_inlet = relative_inlet_radius, k_inlet * relative_inlet_radius
        O_outlet = 1 - relative_outlet_radius, -k_outlet * relative_outlet_radius

        # точки пересечения линий спинки и корыта
        xcl_u, ycl_u = coordinate_intersection_lines(
            (tan(atan(k_inlet) + g_u_inlet), -1,
             sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * relative_inlet_radius -
             (tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (tan(atan(k_outlet) - g_u_outlet), -1,
             sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * relative_outlet_radius -
             (tan(atan(k_outlet) - g_u_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        xcl_d, ycl_d = coordinate_intersection_lines(
            (tan(atan(k_inlet) - g_d_inlet), -1,
             -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * relative_inlet_radius -
             (tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (tan(atan(k_outlet) + g_d_outlet), -1,
             -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * relative_outlet_radius -
             (tan(atan(k_outlet) + g_d_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        # точки пересечения окружностей со спинкой и корытом
        xclc_i_u, yclc_i_u = coordinate_intersection_lines(
            (tan(atan(k_inlet) + g_u_inlet), -1,
             sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * relative_inlet_radius
             - (tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (-1 / (tan(atan(k_inlet) + g_u_inlet)), -1,
             -(-1 / tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1]))

        xclc_i_d, yclc_i_d = coordinate_intersection_lines(
            (tan(atan(k_inlet) - g_d_inlet), -1,
             -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * relative_inlet_radius
             - (tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (-1 / (tan(atan(k_inlet) - g_d_inlet)), -1,
             -(-1 / tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1]))

        xclc_e_u, yclc_e_u = coordinate_intersection_lines(
            (tan(atan(k_outlet) - g_u_outlet), -1,
             sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * relative_outlet_radius
             - tan(atan(k_outlet) - g_u_outlet) * O_outlet[0] - (-1) * O_outlet[1]),
            (-1 / tan(atan(k_outlet) - g_u_outlet), -1,
             -(-1 / tan(atan(k_outlet) - g_u_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        xclc_e_d, yclc_e_d = coordinate_intersection_lines(
            (tan(atan(k_outlet) + g_d_outlet), -1,
             -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * relative_outlet_radius
             - tan(atan(k_outlet) + g_d_outlet) * O_outlet[0] - (-1) * O_outlet[1]),
            (-1 / tan(atan(k_outlet) + g_d_outlet), -1,
             -(-1 / tan(atan(k_outlet) + g_d_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        x, y = list(), list()

        # окружность выходной кромки спинки
        an = angle_between(points=((1, O_outlet[1]), O_outlet, (xclc_e_u, yclc_e_u)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if O_outlet[0] > xclc_e_u: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, discreteness, endpoint=False)
            x += (1 - relative_outlet_radius * (1 - cos(angles))).tolist()
            y += (O_outlet[1] + relative_outlet_radius * sin(angles)).tolist()

        # спинка
        xu, yu = bernstein_curve(((xclc_e_u, yclc_e_u), (xcl_u, ycl_u), (xclc_i_u, yclc_i_u)),
                                 N=discreteness).T.tolist()
        x += xu
        y += yu

        # точки входной окружности кромки по спинке
        an = angle_between(points=((0, O_inlet[1]), O_inlet, (xclc_i_u, yclc_i_u)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if xclc_i_u > O_inlet[0]: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, discreteness, endpoint=False)
            x += (relative_inlet_radius * (1 - cos(angles))).tolist()[::-1]
            y += (O_inlet[1] + relative_inlet_radius * sin(angles)).tolist()[::-1]

        x.pop(), y.pop()  # удаление дубликата входной точки

        # окружность входной кромки корыта
        an = angle_between(points=((0, O_inlet[1]), O_inlet, (xclc_i_d, yclc_i_d)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if xclc_i_d > O_inlet[0]: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, discreteness, endpoint=False)
            x += (relative_inlet_radius * (1 - cos(angles))).tolist()
            y += (O_inlet[1] - relative_inlet_radius * sin(angles)).tolist()

        # корыто
        xd, yd = bernstein_curve(((xclc_i_d, yclc_i_d), (xcl_d, ycl_d), (xclc_e_d, yclc_e_d)),
                                 N=discreteness).T.tolist()
        x += xd
        y += yd

        # точки выходной окружности кромки по корыту
        an = angle_between(points=((1, O_outlet[1]), O_outlet, (xclc_e_d, yclc_e_d)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if O_outlet[0] > xclc_e_d: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, discreteness, endpoint=False)
            x += (1 - relative_outlet_radius * (1 - cos(angles))).tolist()[::-1]
            y += (O_outlet[1] - relative_outlet_radius * sin(angles)).tolist()[::-1]

        return tuple(((x, y) for x, y in zip(x, y)))

    def __mynk(self, discreteness: int,
               mynk_coefficient) -> tuple[tuple[float, float], ...]:

        def mynk_coordinates(param: float, x) -> tuple:
            """Координата y спинки и корыта"""
            part1, part2 = 0.25 * (-x - 17 * x ** 2 - 6 * x ** 3), x ** 0.87 * (1 - x) ** 0.56
            return param * (part1 + part2), param * (part1 - part2)

        x = linspace(0, 1, discreteness, endpoint=True)
        yu, yl = mynk_coordinates(mynk_coefficient, x)
        idx = np.argmax(yu)
        angle = atan((yu[-1] - yu[0]) / (x[-1] - x[0]))

        X, Y = np.hstack((x[-1:idx:-1], x[idx::-1], x[1::])), np.hstack((yu[-1:idx:-1], yu[idx::-1], yl[1::]))

        coordinates = self.transform(tuple(((x, y) for x, y in zip(X, Y))), angle=angle)  # поворот
        x, _ = array(coordinates).T
        return self.transform(coordinates, x0=x.min(), scale=(1 / (x.max() - x.min())))  # нормализация

    def __parsec(self, discreteness: int,
                 relative_inlet_radius,
                 x_relative_camber_upper, x_relative_camber_lower,
                 relative_camber_upper, relative_camber_lower,
                 d2y_dx2_upper, d2y_dx2_lower,
                 theta_outlet_upper, theta_outlet_lower) -> tuple[tuple[float, float], ...]:
        """
        Generate and plot the contour of an airfoil using the PARSEC parameterization
        H. Sobieczky, *'Parametric airfoils and wings'* in *Notes on Numerical Fluid Mechanics*, Vol. 68, pp 71-88]
        (www.as.dlr.de/hs/h-pdf/H141.pdf)
        Repository & documentation: http://github.com/dqsis/parsec-airfoils
        """

        def parsec_coefficients(surface: str,
                                radius_inlet: float | int,
                                c_b: tuple[float, float], d2y_dx2_surface,
                                outlet: tuple, theta_outlet_surface: float | int):
            """PARSEC coefficients"""
            assert surface in ('l', 'u')
            assert isinstance(c_b, (tuple, list)) and len(c_b) == 2
            assert isinstance(outlet, (tuple, list)) and len(outlet) == 2

            x_c_b, y_c_b = c_b
            x_outlet, y_outlet = outlet

            coefs = zeros(6)

            # 1-ый коэффициент зависит от кривой поверхности спинки или корыта
            coefs[0] = -sqrt(2 * radius_inlet) if surface == 'l' else sqrt(2 * radius_inlet)

            i = arange(1, 6)

            # матрицы коэффициентов системы уравнений
            A = array([x_outlet ** (i + 0.5),
                       x_c_b ** (i + 0.5),
                       (i + 0.5) * x_outlet ** (i - 0.5),
                       (i + 0.5) * x_c_b ** (i - 0.5),
                       (i ** 2 - 0.25) * x_c_b ** (i - 1.5)])
            B = array([[y_outlet - coefs[0] * sqrt(x_outlet)],
                       [y_c_b - coefs[0] * sqrt(x_c_b)],
                       [tan(theta_outlet_surface) - 0.5 * coefs[0] * (1 / sqrt(x_outlet))],
                       [-0.5 * coefs[0] * (1 / sqrt(x_c_b))],
                       [d2y_dx2_surface + 0.25 * coefs[0] * x_c_b ** (-1.5)]])

            X = np.linalg.solve(A, B)  # решение СЛАУ
            coefs[1:6] = X[0:5, 0]  # 0 коэффициент уже есть

            return coefs

        # поверхностные коэффициенты давления спинки и корыта
        coefs_u = parsec_coefficients('u', relative_inlet_radius,
                                      (x_relative_camber_upper, relative_camber_upper), d2y_dx2_upper,
                                      (1, 0), theta_outlet_upper)
        coefs_l = parsec_coefficients('l', relative_inlet_radius,
                                      (x_relative_camber_lower, relative_camber_lower), d2y_dx2_lower,
                                      (1, 0), theta_outlet_lower)

        x = linspace(0, 1, discreteness, endpoint=True)

        X, Y = np.hstack((x[::-1], x[1::])), np.hstack((sum([coefs_u[i] * x ** (i + 0.5) for i in range(6)])[::-1],
                                                        sum([coefs_l[i] * x ** (i + 0.5) for i in range(6)])[1::]))
        return tuple((x, y) for x, y in zip(X, Y))

    def __bezier(self, discreteness: int,
                 points) -> tuple[tuple[float, float], ...]:
        X, Y = bernstein_curve(points, N=discreteness).T
        xargmin, xargmax = np.argmin(X), np.argmax(X)
        angle = atan((Y[xargmax] - Y[xargmin]) / (X[xargmax] - X[xargmin]))  # угол поворота
        coordinates = self.transform(tuple(((x, y) for x, y in zip(X, Y))), angle=angle)  # поворот
        x, y = array(coordinates).T
        coordinates = self.transform(coordinates, x0=x.min(), y0=y[0], scale=(1 / (x.max() - x.min())))  # нормализация
        return coordinates

    def __manual(self, discreteness: int,
                 points, deg) -> tuple[tuple[float, float], ...]:
        coordinates = array(points, dtype='float64')
        for _ in range(3):  # для однозначности поворота
            X, Y = array(coordinates, dtype='float64').T
            xargmin, xargmax = np.argmin(X), np.argmax(X)
            angle = atan((Y[xargmax] - Y[xargmin]) / (X[xargmax] - X[xargmin]))  # угол поворота
            coordinates = self.transform(tuple(((x, y) for x, y in zip(X, Y))), angle=angle)  # поворот
        x, y = array(coordinates).T
        coordinates = self.transform(coordinates,
                                     x0=x.min(), y0=y[np.argmin(x)], scale=(1 / (x.max() - x.min())))  # нормализация

        upper_lower = self.upper_lower(coordinates)
        (xu, yu), (xl, yl) = array(upper_lower['upper']).T, array(upper_lower['lower']).T
        fu = interpolate.interp1d(xu, yu, kind=deg, fill_value='extrapolate')
        fl = interpolate.interp1d(xl, yl, kind=deg, fill_value='extrapolate')
        epsrel = 0.000_1
        limit = int(ceil(1 / epsrel))  # предел дискретизации точек интегрирования
        coordinates = list()
        for f, is_upper in zip((fu, fl), (True, False)):
            l = integrate.quad(lambda x: sqrt(1 + derivative(f, x) ** 2), 0, 1, epsrel=epsrel, limit=limit)[0]
            step = l / discreteness
            x = [0]
            while True:  # увеличение дискретизации
                X = x[-1] + step * tan2cos(derivative(f, x[-1]))
                if X > 1: break
                x.append(X)
            X = array(x + [1], dtype='float64')
            if is_upper:
                coordinates += [(x, y) for x, y in zip(X[::-1], f(X[::-1]))]
            else:
                coordinates += [(x, y) for x, y in zip(X[1::], fl(X[1::]))]

        return tuple(coordinates)

    def __circle(self, discreteness: int,
                 relative_circles, rotation_angle, x_ray_cross, is_airfoil) -> tuple[tuple[float, float], ...]:

        y_ray_cross = fsolve(lambda y:
                             atan(x_ray_cross / y) + atan((1 - x_ray_cross) / y) - pi + rotation_angle,
                             array(0.5, dtype='float64'))[0]

        # точки безье средней линии, на которые будут накладываться окружности
        x_av, y_av = bernstein_curve(((0, 0), (x_ray_cross, y_ray_cross), (1, 0)), N=discreteness).T
        f_av = interpolate.interp1d(x_av, y_av, kind=3, fill_value='extrapolate')  # функция средней линии

        # длина кривой центров окружностей
        l = integrate.quad(lambda x: sqrt(1 + derivative(f_av, x) ** 2), 0, 1, epsrel=0.000_1)[0]
        step = l / discreteness  # шаг по кривой центров окружностей

        xc, dc = array(relative_circles, dtype='float64').T

        if is_airfoil:
            xc = np.insert(xc, 0, 0)  # профиль в отличие от канала должен быть замкнутым с концов
            dc = np.insert(dc, 0, 0)  # а для канала в точках х 0 и 1 неизвестны окружности
            xc = np.append(xc, 1)
            dc = np.append(dc, 0)

        xc *= l  # масштабирование хc по длине l

        # непрерывная функция диаметров окружностей по длине средней линии
        f_d = interpolate.interp1d(xc, dc, kind=1, fill_value='extrapolate')  # kind=1 для безопасности

        dy_dx, x_circle, X = list(), list(), -0.5
        while X <= 1.5:
            x_circle.append(X)
            dy_dx.append(derivative(f_av, X))
            X = x_circle[-1] + step * tan2cos(dy_dx[-1])
        x_circle = array(x_circle, dtype='float64')
        y_circle = f_av(x_circle)
        d_circle = np.maximum(f_d(x_circle), 0)  # интерполяция ушла в минус
        dy_dx_ = -1 / array(dy_dx, dtype='float64')  # перпендикуляры

        mask = dy_dx_ >= 0  # маска положительного наклона перпендикуляра
        xu, yu, xl, yl = x_circle.copy(), y_circle.copy(), x_circle.copy(), y_circle.copy()
        d_2_tan2cos, d_2_tan2sin = d_circle / 2 * tan2cos(dy_dx_), d_circle / 2 * tan2sin(dy_dx_)

        xu[mask] += d_2_tan2cos[mask]
        yu[mask] += d_2_tan2sin[mask]
        xl[mask] -= d_2_tan2cos[mask]
        yl[mask] -= d_2_tan2sin[mask]

        xu[~mask] -= d_2_tan2cos[~mask]
        yu[~mask] -= d_2_tan2sin[~mask]
        xl[~mask] += d_2_tan2cos[~mask]
        yl[~mask] += d_2_tan2sin[~mask]

        upper_mask, lower_mask = (0 < xu) & (xu < 1), (0 < xl) & (xl < 1)  # маска принадлежности к интервалу (0, 1)
        xu, yu, xl, yl = xu[upper_mask], yu[upper_mask], xl[lower_mask], yl[lower_mask]

        if not is_airfoil:
            yu -= self.relative_step  # перенос спинки вниз
            yu, yl, xu, xl = yl, yu, xl, xu  # правильное обозначение

        X = np.hstack([[1], xu[::-1], [0], xl, [1]])
        Y = np.hstack([[0], yu[::-1], [0], yl, [0]]) if is_airfoil \
            else np.hstack([[- self.relative_step / 2],
                            yu[::-1],
                            [- self.relative_step / 2],
                            yl,
                            [- self.relative_step / 2]])

        return tuple((x, y) for x, y in zip(X, Y))

    def __fit(self) -> tuple[tuple[float, float], ...]:
        """Профилирование"""
        if self.method == 'NACA':
            self.__coordinates0 = self.__naca(self.discreteness, **self.parameters)
        elif self.method == 'BMSTU':
            self.__coordinates0 = self.__bmstu(self.discreteness, **self.parameters)
        elif self.method == 'MYNK':
            self.__coordinates0 = self.__mynk(self.discreteness, **self.parameters)
        elif self.method == 'PARSEC':
            self.__coordinates0 = self.__parsec(self.discreteness, **self.parameters)
        elif self.method == 'BEZIER':
            self.__coordinates0 = self.__bezier(self.discreteness, **self.parameters)
        elif self.method == 'MANUAL':
            self.__coordinates0 = self.__manual(self.discreteness, **self.parameters)
        elif self.method == 'CIRCLE':
            self.__coordinates0 = self.__circle(self.discreteness, **self.parameters)
        else:
            print(Fore.RED + f'No such method {self.method}! Use Airfoil.help' + Fore.RESET)

        self.__coordinates = self.transform(self.__coordinates0, angle=self.__installation_angle)  # поворот
        coordinates = array(self.__coordinates, dtype='float64').T
        x_min, x_max = coordinates[0].min(), coordinates[0].max()
        self.__chord = abs(x_max - x_min)  # длина хорды
        self.__coordinates = self.transform(self.__coordinates, x0=x_min, scale=(1 / self.__chord))  # нормализация
        return self.__coordinates

    @staticmethod
    def transform(coordinates: tuple[tuple[float, float], ...],
                  x0=0.0, y0=0.0, angle=0.0, scale=1.0) -> tuple[tuple[float, float], ...]:
        """Перенос-поворот-масштабирование кривых спинки и корыта профиля"""
        new_coordinates = list()
        for x, y in coordinates:
            point = Axis.transform(x, y, x0=x0, y0=y0, angle=angle, scale=scale)
            new_coordinates.append((float(point[0]), float(point[1])))
        return tuple(new_coordinates)

    @staticmethod
    def upper_lower(coordinates: tuple[tuple[float, float], ...]) -> dict[str:tuple[tuple[float, float], ...]]:
        """Разделение координат на спинку и корыто"""
        X, Y = array(coordinates, dtype='float64').T
        argmin, argmax = np.argmin(X), np.argmax(X)
        upper, lower = list(), list()
        if argmin < argmax:
            for x, y in zip(X[argmax:-1:+1], Y[argmax:-1:+1]): upper.append((float(x), float(y)))
            for x, y in zip(X[:argmin + 1:+1], Y[:argmin + 1:+1]): upper.append((float(x), float(y)))
            for x, y in zip(X[argmin:argmax + 1:+1], Y[argmin:argmax + 1:+1]): lower.append((float(x), float(y)))
        else:
            for x, y in zip(X[argmax:argmin + 1:+1], Y[argmax:argmin + 1:+1]): upper.append((float(x), float(y)))
            for x, y in zip(X[argmin:-1:+1], Y[argmin:-1:+1]): lower.append((float(x), float(y)))
            for x, y in zip(X[:argmax + 1:+1], Y[:argmax + 1:+1]): lower.append((float(x), float(y)))
        # if upper[-1][0] != 0: upper.append((0, ?)) # неизвестен y входной кромки
        return {'upper': tuple(upper[::-1]), 'lower': tuple(lower)}

    def show(self, amount: int = 2, figsize=(12, 10), savefig=False):
        """Построение профиля"""
        assert isinstance(amount, int) and 1 <= amount  # количество профилей

        X, Y = array(self.coordinates, dtype='float32').T  # запуск расчета
        coordinates0 = self.upper_lower(self.__coordinates0)
        x, y, d, r = self.channel.T

        fg = plt.figure(figsize=figsize)
        gs = fg.add_gridspec(nrows=2, ncols=3)
        plt.suptitle(self.name, fontsize=16, fontweight='bold')

        fg.add_subplot(gs[0, 0])
        plt.title('Initial data')
        plt.axis('off')
        plt.plot([], label=f'method = {self.method}')
        plt.plot([], label=f'discreteness = {self.__discreteness}')
        plt.plot([], label=f'relative_step = {self.__relative_step:.{Foil.__RND}f} []')
        plt.plot([],
                 label=f'gamma = {self.__installation_angle:.{Foil.__RND}f} [rad] = {degrees(self.__installation_angle):.{Foil.__RND}f} [deg]')
        '''for key, value in self.__dict__.items():
            if not key.startswith('_') and isinstance(value, (int, float, np.number)):
                plt.plot([], label=f'{key} = {value:.{Airfoil.__rnd}f}')'''
        plt.legend(loc='upper center')

        fg.add_subplot(gs[1, 0])
        plt.title('Properties')
        plt.axis('off')
        for key, value in self.properties.items(): plt.plot([], label=f'{key} = {value:.{Foil.__RND}f}')
        plt.legend(loc='upper center')

        fg.add_subplot(gs[0, 1])
        plt.title('Airfoil structure')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.plot(*array(coordinates0['upper'], dtype='float16').T, ls='solid', color='blue', linewidth=2)
        plt.plot(*array(coordinates0['lower'], dtype='float16').T, ls='solid', color='red', linewidth=2)

        fg.add_subplot(gs[1, 1])
        plt.title('Channel')
        plt.grid(True)
        plt.ylim([-self.__relative_step / 2, self.__relative_step / 2])
        plt.plot(r, d / 2, ls='solid', color='green', label='channel')
        plt.plot(r, -d / 2, ls='solid', color='green')
        plt.plot([0, max(r)], [0, 0], ls='dashdot', color='orange', linewidth=1.5)
        plt.plot((r[:-1] + r[1:]) / 2, np.diff(d) / np.diff(r), ls='solid', color='red', label='df/dx')
        plt.axis('equal')  # square
        plt.legend(fontsize=12)

        fg.add_subplot(gs[:, 2])
        plt.title('Lattice')
        plt.grid(True)
        plt.axis('equal')  # xlim не нужен ввиду эквивалентности
        plt.xlim([0, 1])
        plt.plot((0, 0), (np.max(Y), np.min(Y) - (amount - 1) * self.__relative_step),
                 (1, 1), (np.max(Y), np.min(Y) - (amount - 1) * self.__relative_step),
                 ls='solid', color='black')  # границы решетки
        for n in range(amount): plt.plot(X, Y - n * self.__relative_step, ls='solid', color='black', linewidth=2)
        alpha = linspace(0, 2 * pi, 360, dtype='float16')
        for i in range(len(d)):
            plt.plot(list(d[i] / 2 * cos(alpha) + x[i]), list(d[i] / 2 * sin(alpha) + y[i]), ls='solid', color='green')
        plt.plot(x, y, ls='dashdot', color='orange')

        plt.tight_layout()
        if savefig: plt.savefig(f'pictures/airfoil_{self.name}.png')
        plt.show()

    @property
    @timeit()
    def properties(self, epsrel: float = 1e-4) -> dict[str: float]:
        if self.__properties: return self.__properties

        limit = int(ceil(1 / epsrel))  # предел дискретизации точек интегрирования
        fu, fl = self.function_upper(3), self.function_lower(3)

        self.__properties['area'] = integrate.dblquad(lambda _, __: 1,
                                                      0, 1, lambda xu: fl(xu), lambda xl: fu(xl),
                                                      epsrel=epsrel)[0]
        x = linspace(0, 1, int(ceil(1 / epsrel)), endpoint=True)
        delta_f = fu(x) - fl(x)
        delta_f_2 = delta_f / 2
        argmax_c, argmax_f = np.argmax(delta_f), np.argmax(np.abs(delta_f_2))
        self.__properties['xc'], self.__properties['c'] = x[argmax_c], delta_f[argmax_c]
        self.__properties['xf'], self.__properties['f'] = x[argmax_f], delta_f_2[argmax_f]  # TODO неверно!
        self.__properties['Sx'] = integrate.dblquad(lambda y, _: y,
                                                    0, 1, lambda xu: fl(xu), lambda xd: fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Sy'] = integrate.dblquad(lambda _, x: x,
                                                    0, 1, lambda xu: fl(xu), lambda xd: fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['x0'] = self.__properties['Sy'] / self.__properties['area'] \
            if self.__properties['area'] != 0 else inf
        self.__properties['y0'] = self.__properties['Sx'] / self.__properties['area'] \
            if self.__properties['area'] != 0 else inf
        self.__properties['Jx'] = integrate.dblquad(lambda y, _: y ** 2,
                                                    0, 1, lambda xu: fl(xu), lambda xd: fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Jy'] = integrate.dblquad(lambda _, x: x ** 2,
                                                    0, 1, lambda xu: fl(xu), lambda xd: fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Jxy'] = integrate.dblquad(lambda y, x: x * y,
                                                     0, 1, lambda xu: fl(xu), lambda xd: fu(xd),
                                                     epsrel=epsrel)[0]
        self.__properties['Jxc'] = self.__properties['Jx'] - self.__properties['area'] * self.__properties['y0'] ** 2
        self.__properties['Jyc'] = self.__properties['Jy'] - self.__properties['area'] * self.__properties['x0'] ** 2
        self.__properties['Jxcyc'] = (self.__properties['Jxy'] -
                                      self.__properties['area'] * self.__properties['x0'] * self.__properties['y0'])
        self.__properties['Jp'] = self.__properties['Jxc'] + self.__properties['Jyc']
        self.__properties['Wp'] = self.__properties['Jp'] / max(
            sqrt((0 - self.__properties['x0']) ** 2 + (0 - self.__properties['y0']) ** 2),
            sqrt((1 - self.__properties['x0']) ** 2 + (0 - self.__properties['y0']) ** 2))
        self.__properties['alpha'] = 0.5 * atan(-2 * self.__properties['Jxcyc'] /
                                                (self.__properties['Jxc'] - self.__properties['Jyc'])) \
            if (self.__properties['Jxc'] - self.__properties['Jyc']) != 0 else -pi / 4
        self.__properties['len_u'] = integrate.quad(lambda x: sqrt(1 + derivative(fu, x) ** 2),
                                                    0, 1, epsrel=epsrel, limit=limit)[0]
        self.__properties['len_l'] = integrate.quad(lambda x: sqrt(1 + derivative(fl, x) ** 2),
                                                    0, 1, epsrel=epsrel, limit=limit)[0]
        return self.__properties

    @property
    @timeit()
    def channel(self) -> np.ndarray:
        """Диффузорность/конфузорность решетки"""
        if len(self.__channel) > 1: return self.__channel

        fu, fl = self.function_upper(3), self.function_lower(3)

        Fu = lambda x: fu(x) - self.__relative_step
        step = self.properties['len_l'] / self.__discreteness  # шаг вдоль кривой

        xgmin, xgmax = 0, 1

        x = [xgmin]
        while True:
            X = x[-1] + step * tan2cos(derivative(fl, x[-1]))
            if X > xgmax: break
            x.append(X)
        x = array(x + [xgmax], dtype='float64')

        Au, _, Cu = coefficients_line(func=fl, x0=x)

        def equations(vars, *args):
            """СНЛАУ"""
            x0, y0, r0, xl = vars
            xu, yu, Au, Cu = args

            Al, _, Cl = coefficients_line(func=Fu, x0=xl)

            return [abs(Au * x0 + (-1) * y0 + Cu) / sqrt(Au ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xu - x0) ** 2 + (yu - y0) ** 2) - r0 ** 2,  # до кривой корыта
                    abs(Al * x0 + (-1) * y0 + Cl) / sqrt(Al ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xl - x0) ** 2 + (Fu(xl) - y0) ** 2) - r0 ** 2]  # до кривой спинки

        if 0:
            fig = plt.figure()
            plt.title('channel')
            plt.axis('equal')
            x = linspace(0, 1, 1000)
            plt.plot(x, fu(x))
            plt.plot(x, fl(x))
            plt.plot(x, Fu(x))
            plt.show()

        xd, yd, d = list(), list(), list()

        warnings.filterwarnings('error')
        for xu, yu, a_u, c_u in tqdm(zip(x, fl(x), Au, Cu), desc='Channel calculation', total=len(x)):
            try:
                res = fsolve(equations, array((xu, yu, self.__relative_step / 2, xu)), args=(xu, yu, a_u, c_u))
            except Exception as exception:
                continue

            if all((xgmin <= res[0] <= xgmax,
                    Fu(res[0]) < res[1] < fl(res[0]),  # y центра окружности лежит в канале
                    xgmin <= res[3] <= xgmax,
                    res[2] * 2 <= self.__relative_step)):
                xd.append(res[0])
                yd.append(res[1])
                d.append(res[2] * 2)
        warnings.filterwarnings('default')

        r = zeros(len(d), dtype='float64')
        for i in range(1, len(d)): r[i] = r[i - 1] + distance((xd[i - 1], yd[i - 1]), (xd[i], yd[i]))

        self.__channel = array((xd, yd, d, r), dtype='float64').T

        return self.__channel

    # TODO
    def cfd(self, vx, vy, padding=0.2, xlim=None, ylim=None):
        """Продувка"""
        assert isinstance(vx, (int, float, np.number))
        assert isinstance(vy, (int, float, np.number))
        assert isinstance(padding, (float, int)) and 0 <= padding

        def u(xy: tuple, vortexs, bounds: tuple = (1, 1)):
            X, Y = xy
            ux = np.full_like(X, bounds[0], dtype=np.float64)
            uy = np.full_like(Y, bounds[1], dtype=np.float64)

            for i, j, k in tqdm(vortexs, desc='CFD'):
                R = ((X - i) ** 2 + (Y - j) ** 2)
                ux += -k * (Y - j) / R
                uy += k * (X - i) / R

            return ux, uy

        x, y = array(self.coordinates).T
        upper_lower = self.upper_lower(self.coordinates)
        upper, lower = upper_lower['upper'], upper_lower['lower']

        vortexs = array((x, y,
                         [-0.5] * (len(x) // 2) + [0.5] * (len(x) - len(x) // 2),  # np.random.randn(len(x))/3
                         )).T

        if xlim is None:
            width = x.max() - x.min()
            xlim = (x.min() - padding * width, x.max() + padding * width)
        if ylim is None:
            height = y.max() - y.min()
            ylim = (y.min() - padding * height, y.max() + padding * height)

        X, Y = np.meshgrid(linspace(*xlim, self.__discreteness * 2), linspace(*ylim, self.__discreteness ** 2))
        ux, uy = u((X, Y), vortexs, bounds=(vx, vy))
        '''for i, ux_ in enumerate(ux):
            for j, uy_ in enumerate(uy):
                if array(upper_lower['lower']) <= uy <= upper_lower['upper']:
                    ux[i], uy[j] = 0, 0'''

        plt.figure(dpi=150)
        plt.streamplot(X, Y, ux, uy,
                       color=(0, 0, 1, 0.5), density=1.5, minlength=0.1, linewidth=0.8, broken_streamlines=True)
        plt.plot(x, y, color='black', linewidth=2)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def test() -> None:
    """Тестирование"""

    Foil.rnd = 4
    print(f'{Foil.rnd = }')

    foils = list()

    if 'NACA' != '':
        parameters = {
            'relative_thickness': 0.2,
            'x_relative_camber': 0.3,
            'relative_camber': 0.05,
            'closed': True, }

        foils.append(Foil('NACA', 40, 1 / 1.698, radians(46.23), name='NACA',
                          **parameters))

    if 'BMSTU' != '':
        parameters = {
            'rotation_angle': radians(70),
            'relative_inlet_radius': 0.06, 'relative_outlet_radius': 0.03,
            'inlet_angle': radians(20), 'outlet_angle': radians(10),
            'x_ray_cross': 0.4,
            'upper_proximity': 0.5, }

        foils.append(Foil('BMSTU', 30, 1 / 1.698, radians(46.23), name='BMSTU',
                          **parameters))

    if 'MYNK' != '':
        parameters = {'mynk_coefficient': 0.2, }

        foils.append(Foil('MYNK', 20, 1 / 1.698, radians(46.23), name='MYNK',
                          **parameters))

    if 'PARSEC' != '':
        parameters = {
            'relative_inlet_radius': 0.06,
            'x_relative_camber_upper': 0.25, 'x_relative_camber_lower': 0.35,
            'relative_camber_upper': 0.1, 'relative_camber_lower': -0.05,
            'd2y_dx2_upper': -0.85, 'd2y_dx2_lower': -0.06,
            'theta_outlet_upper': radians(-6), 'theta_outlet_lower': radians(3), }

        foils.append(Foil('PARSEC', 50, 1 / 1.698, radians(46.23), name='PARSEC',
                          **parameters))

    if 'BEZIER' != '':
        parameters = {'points': ((1.0, 0.0), (0.35, 0.200), (0.05, 0.100),
                                 (0.0, 0.0),
                                 (0.05, -0.10), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0)), }

        foils.append(Foil('BEZIER', 30, 1 / 1.698, radians(46.23), name='BEZIER',
                          **parameters))

    if 'MANUAL' != '':
        parameters = {'points': ((1.0, 0.0), (0.5, 0.15), (0.35, 0.150), (0.10, 0.110), (0.05, 0.08),
                                 (0.0, 0.0),
                                 (0.05, -0.025), (0.35, -0.025), (0.5, 0.0), (0.8, 0.025), (1.0, 0.0)),
                      'deg': 3, }

        foils.append(Foil('MANUAL', 30, 1 / 1.698, radians(46.23), name='MANUAL',
                          **parameters))

    if 'CIRCLE' != '':
        parameters = {'relative_circles': ((0.1, 0.04),
                                           (0.2, 0.035),
                                           (0.3, 0.03),
                                           (0.4, 0.028),
                                           (0.5, 0.025),
                                           (0.6, 0.02),),
                      'rotation_angle': radians(40),
                      'x_ray_cross': 0.5,
                      'is_airfoil': True, }

        foils.append(Foil('CIRCLE', 60, 0.5, radians(30), name='CIRCLE',
                          **parameters))

    if 'CIRCLE' != '':
        parameters = {'relative_circles': ((0.1, 0.4),
                                           (0.2, 0.4),
                                           (0.3, 0.4),
                                           (0.4, 0.4),
                                           (0.5, 0.4),
                                           (0.6, 0.4),
                                           (0.8, 0.4),
                                           (0.9, 0.4),),
                      'rotation_angle': radians(40),
                      'x_ray_cross': 0.5,
                      'is_airfoil': False, }

        foils.append(Foil('CIRCLE', 60, 0.5, radians(30), name='CIRCLE',
                          **parameters))

    if 'Load' != '':
        foil = Foil('NACA', 40, 1, radians(20),
                    relative_thickness=0.1, x_relative_camber=0.3, relative_camber=0, closed=True)
        coordinates = foil.transform(foil.coordinates, x0=foil.properties['x0'], y0=foil.properties['y0'], scale=5)
        foil = Foil.load(coordinates, deg=1, discreteness=80, name='Load')
        foils.append(foil)

    for foil in foils:
        foil.show()

        print(foil.to_dataframe())

        print(Fore.MAGENTA + 'foil properties:' + Fore.RESET)
        for k, v in foil.properties.items(): print(f'{k}: {v}')

        print(Fore.MAGENTA + 'foil channel:' + Fore.RESET)
        print(f'{foil.channel}')

        foil.cfd(10, 5)

        foil.write('csv')


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
