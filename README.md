# foil
Library for profiling aerodynamic foils according to speified parameters.

## About
Profiling methods:
- BMSTU
- NACA (4-digits)
- MYNK
- Bezier
- Manual
- Circles

## Installation

```bash
pip install -r requirements.txt
# or
pip install --upgrade git+https://github.com/ParkhomenkoDV/foil.git
```

## Usage
```python
from foil import METHODS, Foil

help(Foil)

for method, value in METHODS.items():
    print(method)
    for k, v in value.items():
        print(f'{k}: {v}')
    print()
```

**See tutorial in disk/examples/**

## Project structure
foil/
|--- examples  # tutorial
|--- src/foil  # source code
|--- tests
|--- .gitignore
|--- README.md
|--- requirements.txt
|--- setup.py

