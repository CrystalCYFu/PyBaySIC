# BaySIC

**BaySIC** is a Bayesian calibration for the Arctic sea ice biomarker IP<sub>25</sub> and associated open-water phytoplankton biomarkers (brassicasterol or dinosterol). It can be used to predict the ln(PIP<sub>25</sub>) index from sea ice concentration (SIC), or vice versa. When using this model, please cite:

Fu, C. Y., Osman, M. B., & Aquino-López, M. A. (2025). Bayesian calibration for the Arctic sea ice biomarker IP<sub>25</sub>. _Paleoceanography and Paleoclimatology_, 40, e2024PA005048. https://doi.org/10.1029/2024PA005048

💡 **Prefer MATLAB?** Check out [BaySIC-MATLAB](https://github.com/mattosman/BaySIC-MATLAB)!

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
- Required libraries: Matplotlib, NumPy, Pandas, SciPy

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

baysic = PyBaySIC.BaySIC()
```

---

### 1. Forward Modelling

To predict ln(P<sub>D</sub>IP<sub>25</sub>) or ln(P<sub>B</sub>IP<sub>25</sub>) from SIC, use `forward()` with the following inputs:
1. `sic`: scalar or vector of fractional SIC (0–1)
2. `index`: `'dino'` or `'bras'`

*And optionally:*

3. `mode`: default to `'plot'`
   - `'data'` returns the posterior distribution evaluated over a grid of ln(PIP<sub>25</sub>) values
   - `'summary'` returns the maximum a posteriori (MAP) estimate and HDI limits
4. `hdiMass`: (0-1), default to `(0.15, 0.35, 0.55, 0.75, 0.95)`

For example:

 ```
# Predict ln(PᴅIP₂₅) for SIC=0.92, plot results
baysic.forward(0.92, 'dino')
plt.show()

# Predict ln(PʙIP₂₅) and calculate the 50% HDI, print MAP estimate and HDI limits
sic = (0.2, 0.4, 0.6, 0.8)
results = baysic.forward(sic, 'bras', 'summary', 0.5)
print(results)
```

The `forward()` function uses the **spatially varying (3 months before first SIC decrease) calibration**.

#### 💡 Calculating mean SIC of the 3 months before the first SIC decrease

The `cal_meanSIC()` function determines the required `sic` from a 12 × 1 input array of monthly SIC climatology, `sic_climo` (0-1).

The second output indicates the represented months (`0` = January, `1` = February, ..., `11` = December).

Where SIC remains constant throughout the year, the month of the first SIC decrease should be drawn from the nearest grid with variable SIC. This can be done using the same function with the following inputs:
1. `sic_climo`: a 12 × latitude × longitude spatial field of monthly SIC climatologies
2. `site_lat` and `site_lon`: the latitude and longitude of the target site
3. `all_lat` and `all_lon`: the latitudes and longitudes corresponding to `sic_climo`

#### 💡 Calculating ln(PIP<sub>25</sub>)

The `cal_lnPIP()` function computes ln(PIP<sub>25</sub>) from [IP<sub>25</sub>] and [brassicasterol]/[dinosterol] with the appropriate detection limit treatment, which is also applied in the inverse model.

It takes the same inputs (1-4) as `inverse()` (see below), and its outputs can be directly compared with the forward modelling results.

---

### 2. Inverse Modelling

To predict SIC from [IP<sub>25</sub>] and [brassicasterol]/[dinosterol], use `inverse()` with the following inputs:
1. `ip25`: scalar or vector of IP<sub>25</sub> concentration (>=0)
2. `sterol`: scalar or vector of brassicasterol or dinosterol concentration (>=0), *in the same units as [IP<sub>25</sub>]*
3. `index`: `'dino'` or `'bras'`
4. `unit`: `'toc'` or `'sed'`

*And optionally:*

5. `mode`: default to `'plot'`
   - `'data'` returns the posterior distribution evaluated over a grid of SIC values
   - `'summary'` returns the MAP estimate and HDI limits
6. `hdiMass`: (0-1), default to `(0.15, 0.35, 0.55, 0.75, 0.95)`
7. `xType`: `'age'` or `'depth'`, default to index
8. `xVal`: scalar or vector (in ascending/descending order) of age or depth (>=0), *in ka BP or m*

If either `xType` or `xVal` is provided, the other must also be specified.

For example:

 ```
ip25 = np.random.uniform(0, 0.09, 20)
sterol = np.random.uniform(0, 9, 20)
ages = np.arange(0, 40, 2)

baysic.inverse(ip25, sterol, 'dino', 'toc', xType='age', xVal=ages)
plt.show()
```

The `inverse()` function uses the **Arctic-wide static (March-April-May) calibration**.

## Contributing

Contributions are welcome! If you'd like to report a bug, request a feature, or suggest an improvement, please create a pull request, or open an issue [here](https://github.com/CrystalCYFu/PyBaySIC/issues).

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

**Copyright (c) 2025**
