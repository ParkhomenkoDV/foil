from colorama import Fore
from numpy import radians

from foil import Foil


def main() -> None:
    """Тестирование"""

    foils = list()

    if "NACA" != "":
        parameters = {
            "relative_thickness": 0.2,
            "x_relative_camber": 0.3,
            "relative_camber": 0.05,
            "closed": True,
        }

        foils.append(
            Foil(
                "NACA",
                40,
                chord=0.1,
                installation_angle=radians(30),
                step=0.06,
                name="NACA",
                **parameters,
            )
        )

    if "BMSTU" != "":
        parameters = {
            "rotation_angle": radians(70),
            "relative_inlet_radius": 0.06,
            "relative_outlet_radius": 0.03,
            "inlet_angle": radians(20),
            "outlet_angle": radians(10),
            "x_ray_cross": 0.4,
            "upper_proximity": 0.5,
        }

        foils.append(
            Foil(
                "BMSTU",
                30,
                chord=0.05,
                installation_angle=radians(30),
                step=0.03,
                name="BMSTU",
                **parameters,
            )
        )

    if "MYNK" != "":
        parameters = {
            "mynk_coefficient": 0.2,
        }

        foils.append(
            Foil(
                "MYNK",
                20,
                chord=0.25,
                installation_angle=radians(30),
                step=0.2,
                name="MYNK",
                **parameters,
            )
        )

    if "PARSEC" != "":
        parameters = {
            "relative_inlet_radius": 0.01,
            "x_relative_camber_upper": 0.35,
            "x_relative_camber_lower": 0.45,
            "relative_camber_upper": 0.055,
            "relative_camber_lower": -0.006,
            "d2y_dx2_upper": -0.35,
            "d2y_dx2_lower": -0.2,
            "theta_outlet_upper": radians(-10),
            "theta_outlet_lower": radians(2),
        }

        foils.append(
            Foil(
                "PARSEC",
                50,
                chord=0.06,
                installation_angle=radians(30),
                step=0.06,
                name="PARSEC",
                **parameters,
            )
        )

    if "BEZIER" != "":
        parameters = {
            "coordinates": (
                (1.0, 0.0),
                (0.35, 0.200),
                (0.05, 0.100),
                (0.0, 0.0),
                (0.05, -0.10),
                (0.35, -0.05),
                (0.5, 0.0),
                (1.0, 0.0),
            ),
        }

        foils.append(
            Foil(
                "BEZIER",
                30,
                chord=0.8,
                installation_angle=radians(30),
                step=0.6,
                name="BEZIER",
                **parameters,
            )
        )

    if "MANUAL" != "":
        parameters = {
            "coordinates": (
                (1.0, 0.0),
                (0.5, 0.15),
                (0.35, 0.150),
                (0.10, 0.110),
                (0.05, 0.08),
                (0.0, 0.0),
                (0.05, -0.025),
                (0.35, -0.025),
                (0.5, 0.0),
                (0.8, 0.025),
                (1.0, 0.0),
            ),
            "deg": 3,
        }

        foils.append(
            Foil(
                "MANUAL",
                30,
                chord=1.2,
                installation_angle=radians(30),
                step=0.6,
                name="MANUAL",
                **parameters,
            )
        )

    if "CIRCLE" != "":
        parameters = {
            "relative_circles": (
                (0.1, 0.04),
                (0.2, 0.035),
                (0.3, 0.03),
                (0.4, 0.028),
                (0.5, 0.025),
                (0.6, 0.02),
            ),
            "rotation_angle": radians(40),
            "x_ray_cross": 0.5,
            "is_foil": True,
        }

        foils.append(
            Foil(
                "CIRCLE",
                60,
                chord=0.5,
                installation_angle=radians(30),
                step=0.3,
                name="CIRCLE",
                **parameters,
            )
        )

    if "CIRCLE" != "":
        parameters = {
            "relative_circles": (
                (0.1, 0.4),
                (0.2, 0.4),
                (0.3, 0.4),
                (0.4, 0.4),
                (0.5, 0.4),
                (0.6, 0.4),
                (0.8, 0.4),
                (0.9, 0.4),
            ),
            "rotation_angle": radians(40),
            "x_ray_cross": 0.5,
            "is_foil": False,
        }

        foils.append(
            Foil(
                "CIRCLE",
                60,
                chord=0.6,
                installation_angle=radians(30),
                step=0.5,
                name="CIRCLE",
                **parameters,
            )
        )

    if "Load" != "":
        foil = Foil(
            "NACA",
            40,
            chord=1,
            installation_angle=radians(30),
            step=0.6,
            relative_thickness=0.1,
            x_relative_camber=0.3,
            relative_camber=0,
            closed=True,
        )
        coordinates = foil.transform(
            foil.coordinates,
            transfer=(foil.properties()["x0"], foil.properties()["y0"]),
            scale=5,
        )
        foil = Foil.load(coordinates, deg=1, discreteness=80, step=3, name="Load")
        foils.append(foil)

    for foil in foils:
        foil.plot()  # .show()

        for relative in (True, False):
            print(foil.to_dataframe(relative=relative))

            print(Fore.MAGENTA + "foil properties:" + Fore.RESET)
            for key, value in foil.properties(relative=relative).items():
                print(f"{key}: {value}")

            for extension in ("txt", "csv", "xlsx", "pkl"):
                foil.write(extension, relative=relative)

        print(Fore.MAGENTA + "foil channel:" + Fore.RESET)
        print(f"{foil.channel}")

        # foil.cfx(10, 5)


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()", sort="cumtime")
