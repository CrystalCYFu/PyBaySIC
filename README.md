# BaySIC

**BaySIC** is a Bayesian calibration for the Arctic sea ice biomarker IP<sub>25</sub> and associated open-water phytoplankton biomarkers (brassicasterol or dinosterol). It can be used to predict the ln(PIP<sub>25</sub>) index from sea ice concentration (SIC), or vice versa. When using this model, please cite:

*Citation...*

## Features

- **Nonlinearity**: BaySIC uses an inverse logistic function to characterise the nonlinear relationship between SIC and ln(PIP<sub>25</sub>), respecting the natural limit of SIC between 0 and 1.
- **Bi-directional uncertainty quantification**: Calibration uncertainties are quantified using highest density intervals (HDIs) in the outputs of both the forward and inverse models.
- **Non-stationary seasonality**: The forward model is based on a spatially varying calibration that correlates ln(PIP<sub>25</sub>) with the mean SIC for the three-month interval before the first SIC decrease, accounting for spatiotemporal variations in proxy seasonality.
- **Salinity as an additional environmental driver**: Thresholds have been identified for sea surface salinity below which SIC ceases to be the dominant driver of ln(PIP<sub>25</sub>); we caution against the use of BaySIC in such cases.

For more details, please refer to the source publication.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8
- Required libraries: Matplotlib, NumPy, SciPy, tqdm

### Installation

Clone the repository directly:

```bash
git clone https://github.com/CrystalCYFu/PyBaySIC.git
cd PyBaySIC
```

## Usage

First, load the required packages and create an instance of the BaySIC class:

```bash
import PyBaySIC
import matplotlib.pyplot as plt
import numpy as np

test = PyBaySIC.BaySIC()
```

### Forward Modelling

To predict ln(P<sub>D</sub>IP<sub>25</sub>) or ln(P<sub>B</sub>IP<sub>25</sub>) from SIC, use `forward()` with the following inputs:
1. `sic` (0-1)
2. `index` (`'dino'`/`'bras'`)

*And optionally:*

3. `hdiMass` (0-1), default to `(0.15, 0.35, 0.55, 0.75, 0.95)`

For example:

 ```
sic = 0.92

test.forward(sic, 'dino')
plt.show()
```

The `forward()` function uses the spatially varying (3 months before first SIC decrease) calibration.

### Inverse Modelling

To predict SIC from [IP<sub>25</sub>] and [brassicasterol]/[dinosterol], use `inverse()` with the following inputs:
1. `ip25` (>=0)
2. `sterol` (>=0), *in the same units as [IP<sub>25</sub>]*
3. `index` (`'dino'`/`'bras'`)
4. `unit` (`'toc'`/`'sed'`)

*And optionally:*

5. `hdiMass` (0-1), default to `(0.15, 0.35, 0.55, 0.75, 0.95)`
6. `xType` (`'age'`/`'depth'`), default to index
7. `xVal` (>=0, in ascending/descending order), *age expected in ka BP, depth expected in m*

If either `xType` or `xVal` is provided, the other must also be specified.

For example:

 ```
ip25 = np.random.uniform(0, 0.09, 20)
sterol = np.random.uniform(0, 9, 20)
ages = np.arange(0, 40, 2)

test.inverse(ip25, sterol, 'dino', 'toc', xType='age', xVal=ages)
plt.show()
```

The `inverse()` function uses the Arctic-wide static (March-April-May) calibration.

## Contributing

Contributions are welcome! If you'd like to report a bug, request a feature, or suggest an improvement, please create a pull request, or open an issue [here](https://github.com/CrystalCYFu/PyBaySIC/issues).

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

**Copyright (c) 2025**
