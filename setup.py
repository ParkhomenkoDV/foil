from setuptools import setup, find_packages
import time

with open('README.md', 'rt', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', 'rt') as file:
    install_requires = file.readlines()
    print(install_requires)

setup(
    name='foil',
    version=time.strftime('%Y.%m.%d.%H', time.localtime()),
    description='lib',
    long_description=long_description,
    long_description_content_type='text/markdown',  # если long_description = .md
    author='Daniil Andryushin',
    author_email='parkho.m.enko@mail.ru',
    url='https://github.com/ParkhomenkoDV/foil.git',
    packages=find_packages(where="src", exclude=["tests*", "docs*", "examples*"]),
    package_dir={"": "src"},
    python_requires='>=3.11',
    install_requires=install_requires,
)
