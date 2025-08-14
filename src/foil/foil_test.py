import numpy as np
import pytest
from numpy import pi, radians

from foil import Foil

# Общие параметры для тестов
DEFAULT_DISCRETENESS = 30
DEFAULT_CHORD = 1.0
DEFAULT_INSTALLATION_ANGLE = 0.0
DEFAULT_STEP = 1.0
DEFAULT_START_POINT = (0, 0)


class TestFoil:
    def test_foil_initialization(self):
        """Проверка инициализации профиля."""
        foil = Foil(
            "NACA",
            discreteness=DEFAULT_DISCRETENESS,
            **{
                "relative_thickness": 0.2,
                "x_relative_camber": 0.3,
                "relative_camber": 0.05,
                "closed": True,
            },
        )
        assert foil.method == "NACA"
        assert foil.discreteness == DEFAULT_DISCRETENESS
        assert foil.chord == DEFAULT_CHORD
        assert foil.installation_angle == DEFAULT_INSTALLATION_ANGLE
        assert foil.step == DEFAULT_STEP
        assert foil.start_point == DEFAULT_START_POINT

    def test_foil_properties(self):
        """Проверка свойств профиля."""
        foil = Foil(
            "NACA",
            discreteness=50,
            chord=2.0,
            installation_angle=radians(15),
            step=0.5,
            **{
                "relative_thickness": 0.2,
                "x_relative_camber": 0.3,
                "relative_camber": 0.05,
                "closed": True,
            },
        )
        assert foil.discreteness == 50
        assert foil.chord == 2.0
        assert foil.installation_angle == radians(15)
        assert foil.step == 0.5

    def test_naca_profile(self):
        """Проверка создания профиля NACA."""
        foil = Foil(
            "NACA",
            relative_thickness=0.12,
            x_relative_camber=0.4,
            relative_camber=0.02,
            closed=True,
        )
        assert len(foil.relative_coordinates) > 0
        assert len(foil.coordinates) > 0

    def test_bmstu_profile(self):
        """Проверка создания профиля BMSTU."""
        foil = Foil(
            "BMSTU",
            rotation_angle=radians(70),
            relative_inlet_radius=0.05,
            relative_outlet_radius=0.03,
            inlet_angle=radians(20),
            outlet_angle=radians(10),
            x_ray_cross=0.4,
            upper_proximity=0.5,
        )
        assert len(foil.relative_coordinates) > 0
        assert len(foil.coordinates) > 0

    def test_parsec_profile(self):
        """Проверка создания профиля PARSEC."""
        foil = Foil(
            "PARSEC",
            relative_inlet_radius=0.01,
            x_relative_camber_upper=0.35,
            x_relative_camber_lower=0.45,
            relative_camber_upper=0.05,
            relative_camber_lower=-0.01,
            d2y_dx2_upper=-0.3,
            d2y_dx2_lower=-0.2,
            theta_outlet_upper=radians(-10),
            theta_outlet_lower=radians(5),
        )
        assert len(foil.relative_coordinates) > 0
        assert len(foil.coordinates) > 0

    def test_transform_coordinates(self):
        """Проверка трансформации координат (перенос, поворот, масштабирование)."""
        points = [(0, 0), (1, 0), (0.5, 0.5)]
        transformed = Foil.transform(points, transfer=(1, 1), angle=pi / 2, scale=2)
        expected = [(-2, 2), (-2, 0), (-1, 1)]
        for (x, y), (ex, ey) in zip(transformed, expected):
            assert np.isclose(x, ex), f"{x}!={ex}"
            assert np.isclose(y, ey), f"{y}!={ey}"

    def test_foil_properties_calculation(self):
        """Проверка расчета свойств профиля."""
        foil = Foil(
            "NACA",
            relative_thickness=0.1,
            x_relative_camber=0.3,
            relative_camber=0.02,
            closed=True,
        )
        props = foil.properties(relative=True)
        assert "chord" in props
        assert "area" in props
        assert props["chord"] == 1.0  # Относительная хорда

        abs_props = foil.properties(relative=False)
        assert abs_props["chord"] == foil.chord

    def test_load_and_export(self):
        """Проверка загрузки профиля из координат и экспорта в DataFrame."""
        coordinates = [(1, 0), (0.5, 0.1), (0, 0), (0.5, -0.05)]
        foil = Foil.load(coordinates, deg=1)
        assert len(foil.coordinates) > 0

        df = foil.to_dataframe(relative=True)
        assert len(df) == len(foil.relative_coordinates)
        assert "x" in df.columns and "y" in df.columns

    def test_invalid_method(self):
        """Проверка обработки неверного метода профилирования."""
        with pytest.raises((AssertionError, ValueError)):
            Foil("INVALID_METHOD")

    @pytest.mark.skip
    def test_invalid_parameters():
        """Проверка обработки неверных параметров."""
        with pytest.raises(AssertionError):
            Foil("NACA", relative_thickness=-0.1)  # Отрицательная толщина

    def test_channel_calculation(self):
        """Проверка расчета канала решетки."""
        foil = Foil(
            "NACA",
            relative_thickness=0.1,
            x_relative_camber=0.3,
            relative_camber=0.02,
            closed=True,
            step=0.5,
        )
        channel = foil.channel
        assert "x" in channel
        assert "y" in channel
        assert "d" in channel
        assert "r" in channel
