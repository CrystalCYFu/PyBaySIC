# BaySIC! :D

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required input:
# 1. predictor
#       forward -> sic (0-1)
#       inverse -> ip25 and sterol (>=0), in same units!!
# 2. index ('dino'/'bras')
# 3. unit ('toc'/'sed'), for inverse model only

# Optional input:
# 4. mode ('data'/'summary'/'samples'), default to 'plot'
# 5. hdiMass (0-1), default to (0.15, 0.35, 0.55, 0.75, 0.95)
# 6a. xType ('age'/'depth'), for inverse model only, default to index
# 6b. xVal (>=0, in ascending/descending order), for inverse model only
#        age in ka BP, depth in m

# Calibration interval:
# forward -> 3 months before 1st SIC decrease
# inverse -> MAM

# Output:
# 'plot': kernel distribution (for up to 6 predictions)
#         highest density interval (HDI)
#         maximum a posteriori (MAP) estimate
# 'data': posterior distribution
# 'summary': MAP estimate and HDI limits
# 'samples': 10000 samples randomly drawn from posterior distribution

# Helper functions:
# cal_meanSIC -> calculate mean SIC of 3 months before 1st SIC decrease
#                from monthly SIC climatology
# cal_lnPIP -> calculate lnPIP from IP25 and sterol concentrations
#              w/ detection limit treatment

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

try:
    import os
except ImportError:
    print ("Please install os.")

try:
    import numpy as np
except ImportError:
    print ("Please install NumPy.")

try:
    import pandas as pd
except ImportError:
    print ("Please install Pandas.")

try:
    from scipy.stats import norm
except ImportError:
    print ("Please install SciPy.")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print ("Please install Matplotlib.")


class BaySIC:

    # These values are added to all biomarker inputs in calculating lnPIP
    min_ip25_toc = 0.01049169444307899
    min_ip25_sed = 6.212970072165696e-05
    min_dino_toc = 0.21311343280837622
    min_dino_sed = 0.0008602308083664112
    min_bras_toc = 0.14608475206034371
    min_bras_sed = 0.0005896700301752964

    # Function to initialize object's attributes + perform setup operations
    def __init__(self):

        # Get current directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))


    # Function to ensure predictor/hdiMass/xVal is 1) an array of floats and 2) within valid ranges
    def check_input(self, value, category):

        # Part 1
        if isinstance(value, (int, float)):     # if predictor/hdiMass/xVal is a single value (integer or float)
            value = [value]                     # wrap in list
        
        if isinstance(value, (list, tuple)):    # if it is a list or a tuple
            value = np.array(value)             # convert to array
        elif isinstance(value, (np.ndarray)):   # if it is already an array
            pass                                # do nothing
        else:                                   # raise error for everything else
            raise ValueError(f"Invalid value for {category}: '{value}'.")

        value = value.astype(float)             # cast to float type

        # Part 2
        if category == 'SIC':
            # Check if sic values are between 0 and 1
            if (value<0).any() or (value>1).any():
                raise ValueError("SIC must be between 0 and 1.")
            # Treat sic = 0 or 1
            else:
                value[value==0] += 1e-4
                value[value==1] -= 1e-4

        elif category == 'hdiMass':
            # Check if hdiMass values are between 0 and 1
            if (value<0).any() or (value>1).any():
                raise ValueError("hdiMass must be between 0 and 1.")
            # Sort in ascending order
            else:
                value = np.sort(value)
        
        elif category == 'IP₂₅' or category == 'sterol':
            # Check if concentrations are non-negative
            if (value<0).any():
                raise ValueError(f"{category} must be non-negative.")
            
        elif category == 'age':
            # Check if ages are in ascending/descending order
            if not (np.all(np.diff(value) >= 0) or np.all(np.diff(value) <= 0)):
                raise ValueError(f"xVal ({category}) must be in ascending or descending order.")
            
        elif category == 'depth':
            # Check if depths are non-negative
            if (value<0).any():
                raise ValueError(f"{category} must be non-negative.")
            # Check if depths are in ascending/descending order
            elif not (np.all(np.diff(value) >= 0) or np.all(np.diff(value) <= 0)):
                raise ValueError(f"xVal ({category}) must be in ascending or descending order.")

        return value
    

    # Function to calculate lnPIP from ip25 and sterol
    def cal_lnPIP(self, ip25, sterol, index, unit):
        
        # Ensure ip25 and sterol are arrays of floats and within valid ranges
        ip25 = self.check_input(ip25, 'IP₂₅')
        sterol = self.check_input(sterol, 'sterol')

        # Ensure ip25 and sterol are of same length
        if len(ip25) != len(sterol):
            raise ValueError("The lengths of ip25 and sterol do not match. Please use paired measurements.")
        
        # Normalize input strings to lowercase
        index = str(index).lower()
        unit = str(unit).lower()

        # Check index is valid
        if index not in ['dino', 'bras']:
            raise ValueError(f"Invalid value for index: '{index}'. Please use one of the following: 'dino', 'bras'.")
        
        # Treat biomarker concentrations
        if unit == 'toc':
            ip25 += self.min_ip25_toc
            if index == 'dino':
                sterol += self.min_dino_toc
            else:
                sterol += self.min_bras_toc
        elif unit == 'sed':
            ip25 += self.min_ip25_sed
            if index == 'dino':
                sterol += self.min_dino_sed
            else:
                sterol += self.min_bras_sed
        else:
            raise ValueError(f"Invalid value for unit: '{unit}'. Please use one of the following: 'toc', 'sed'.")

        # Calculate lnPIP
        lnPIP = np.log(ip25/(ip25+sterol))

        return lnPIP


    # Function to find high density region
    # (adapted from https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/HDI_of_grid.py)
    def HDIofGrid(self, probMassVec, credMass):

        # sort probability masses from highest to lowest (::-1 = start at end, step backward 1 element at a time)
        # move down sorted queue until cumulative probability > mass desired
        sortedProbMass = np.sort(probMassVec, axis=None)[::-1]
        HDIheightIdx = np.min(np.where(np.cumsum(sortedProbMass) >= credMass))

        # 'HDIheight' = smallest component probability mass in HDI
        # 'HDImass' = total mass of included indices
        # 'idx' = vector of indices in HDI
        HDIheight = sortedProbMass[HDIheightIdx]
        # HDImass = np.sum(probMassVec[probMassVec >= HDIheight])
        idx = np.where(probMassVec >= HDIheight)

        # Get indices of HDI limits (assume continuous HDI!)
        idx = idx[0]
        lowerHDI = np.min(idx)
        upperHDI = np.max(idx)

        return lowerHDI, upperHDI


    # Function to create subplots (up to 6)
    def create_subplots(self, num_subplots):

        # 2-3 subplots -> 1 row
        if num_subplots < 4:
            fig, axs = plt.subplots(1, num_subplots, figsize=(3*(num_subplots+.3),3))
        
        # 4-6 subplots -> 2 rows
        else:
            num_cols = int(np.ceil(num_subplots/2))   # round up, e.g. 5 subplots -> 3 columns
            fig, axs = plt.subplots(2, num_cols, figsize=(3*(num_cols+.3),6))
            axs = axs.flatten()     # flatten array of axes for easy iteration
            if num_subplots == 5:   # 5 subplots -> delete last
                fig.delaxes(axs[-1])
        
        return fig, axs
    

    # Function to plot results in subplots
    def fill_subplots(self, ax, lowerHDI_idx_list, upperHDI_idx_list, x_grid, hdiMass, mapEstimate, pdf, c, c1):

        # Reverse list of lower HDI limits, extend with upper HDI limits, create pairs
        # E.g. hdiMass = 0.3,0.6,0.9, lower limits (indices) = 30,20,10, upper limits = 40,50,60
        # idx_list = 0,10,20,30,40,50,60,70, idx_pairs = [0,10],[10,20],[20,30],...,[60,70]
        lowerHDI_idx_list = lowerHDI_idx_list[::-1]
        idx_list = [0] + lowerHDI_idx_list + upperHDI_idx_list + [len(x_grid)-1]
        idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

        # Create array of alphas, mirror itself (-2 = start from 2nd last element, ::-1 = reverse array)
        alphas = np.insert(hdiMass, 0, 0.05)
        alphas_m = np.concatenate((alphas, alphas[-2::-1]))
        alphas_m = alphas_m*0.9

        # Reverse hdiMass, add '100% HDI', mirror itself
        hdiMass_r = hdiMass[::-1]
        hdiMass_r = np.insert(hdiMass_r, 0, 1)
        hdiMass_m = np.concatenate((hdiMass_r, hdiMass_r[-2::-1]))

        # Plot MAP estimate as vertical line, add label
        ax.axvline(x=mapEstimate, color=c1, alpha=0.75, label='MAP estimate')
        if mapEstimate < (x_grid[0] + x_grid[-1]) / 2:   # left half of plot
            ax.text(mapEstimate, max(pdf)*1.05, f' {round(mapEstimate,2)}', ha='left', va='center')
        else:   # right half
            ax.text(mapEstimate, max(pdf)*1.05, f'{round(mapEstimate,2)} ', ha='right', va='center')

        # Shade HDI
        k=0
        for pair, alpha, m in zip(idx_pairs, alphas_m, hdiMass_m):

            x = x_grid[pair[0]:pair[1]]
            y = pdf[pair[0]:pair[1]]

            if k < int(len(hdiMass_m)/2):
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None)
            elif k == len(hdiMass_m)-1:
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None, label=f'>{round(hdiMass_m[-2]*100)}% HDI')
            else:
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None, label=f'{round(m*100)}% HDI')

            k+=1

        ax.set_ylim(bottom=0, top=max(pdf)*1.1)
        ax.set_ylabel('P')
        ax.yaxis.set_ticks([])


    # Function to format subplots with forward modelling results (lnPIP from sic)
    def forward_subplots(self, ax, sic_val, label):

        ax.set_xlim(left=-12, right=0)   # lnPIP lower limit...
        ax.set_xlabel(label)
        ax.set_title(f'SIC = {round(sic_val, 3)}')   # if input SIC = 0/1, this shows treated SIC


    # Function to format subplots with inverse modelling results (sic from lnPIP)
    def inverse_subplots(self, ax, lnPIP_val, xVal, xlabel, label):

        ax.set_xlim(left=0, right=1)
        ax.set_xlabel('SIC')

        if xlabel == 'Age (ka BP)':
            ax.set_title(f'{xVal} ka BP, {label} = {round(lnPIP_val, 3)}')
        elif xlabel == 'Depth (m)':
            ax.set_title(f'{xVal} m, {label} = {round(lnPIP_val, 3)}')
        else:   # index not needed
            ax.set_title(f'{label} = {round(lnPIP_val, 3)}')

    
    # Function to plot results in series
    def fill_series(self, ax, idx_pairs_list, y_grid, hdiMass, mapEstimate_list, x, c, c1):

        # Create array of alphas, mirror itself (-2 = start from 2nd last element, ::-1 = reverse array)
        alphas_m = np.concatenate((hdiMass, hdiMass[-2::-1]))
        alphas_m = alphas_m*0.84

        # Reverse hdiMass, mirror itself
        hdiMass_r = hdiMass[::-1]
        hdiMass_m = np.concatenate((hdiMass_r, hdiMass_r[-2::-1]))

        # Convert list to array, switch axis 0 and 1 (sic/lnPIP index, HDI)
        idx_pairs_array = np.array(idx_pairs_list)
        transposed_array = np.transpose(idx_pairs_array, (1, 0, 2))

        # Plot MAP estimates as broken line
        ax.plot(x, mapEstimate_list, color=c1, clip_on=False, label='MAP estimate')

        # Iterate over axis 0 (HDI), shade HDI
        for i, sub_array in enumerate(transposed_array):

            y1 = y_grid[sub_array[:,0]]
            y2 = y_grid[sub_array[:,1]]

            if i < int(len(hdiMass_m)/2):
                ax.fill_between(x, y1, y2, color=c, alpha=alphas_m[i], edgecolor=None)
            else:
                ax.fill_between(x, y1, y2, color=c, alpha=alphas_m[i], edgecolor=None,
                                label=f'{round(hdiMass_m[i]*100)}% HDI')


    # Function to format series of forward modelling results (lnPIP from sic)
    def forward_series(self, ax, sic, x, label):

        ax.set_xlim(left=1, right=len(sic))
        ax.set_ylim(bottom=-12, top=0)   # lnPIP lower limit...

        ax.set_xticks(x)
        ax.set_xlabel('Index')
        ax.set_ylabel(label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


    # Function to format series of inverse modelling results (sic from lnPIP)
    def inverse_series(self, ax, xVal, xlabel, label):

        ax.set_xlim(left=np.min(xVal), right=np.max(xVal))
        ax.set_ylim(bottom=0, top=1)

        if xlabel == 'Index':
            ax.set_xticks(xVal.astype(int))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('SIC')

        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        legend.set_title(f'Predictor: {label}')
        legend._legend_box.align = 'left'


    # Function to forward model lnPIP from sic
    def forward(self, sic, index, mode='plot', hdiMass=(0.15,0.35,0.55,0.75,0.95)):

        # Ensure sic and hdiMass are arrays of floats and within valid ranges
        sic = self.check_input(sic, 'SIC')
        hdiMass = self.check_input(hdiMass, 'hdiMass')

        # Normalize input string to lowercase
        index = str(index).lower()

        # Select '3 months before 1st sic loss' posteriors, colours and labels
        if index == 'dino':
            filename = 'server_dino_spavar_7900_2024-09-06_08-56-41.txt'
            c = 'tab:red'
            c1 = 'darkred'
            label = r'$\ln(\mathrm{P_{D}IP_{25}})$'
        elif index == 'bras':
            filename = 'server_bras_spavar_7900_2024-09-06_08-56-41.txt'
            c= 'tab:blue'
            c1 = 'darkblue'
            label = r'$\ln(\mathrm{P_{B}IP_{25}})$'
        else:
            raise ValueError(f"Invalid value for index: '{index}'. Please use one of the following: 'dino', 'bras'.")
        
        # Construct path to file containing posterior, read file
        posterior_path = os.path.join(self.current_dir, filename)
        posterior = np.genfromtxt(posterior_path, delimiter='\t')

        # Load regression coefficients and precision
        b0 = posterior[:, 0]
        b1 = posterior[:, 1]
        phi = posterior[:, 2]

        # Calculate standard deviation
        sd = np.sqrt(phi)

        # Create evenly spaced grid within expected range of lnPIP - lower limit...
        lnPIP_grid = np.linspace(-12, 0, 1000)

        # For up to 6 sic values, create figure to plot PDF, HDI, and MAP estimate for each
        # Otherwise, create lists to save results for plotting later
        if mode == 'plot':
            if len(sic) == 1:
                fig, axs = plt.subplots(figsize=(4.2,3))
            elif len(sic) <= 6:
                fig, axs = self.create_subplots(len(sic))
            else:
                idx_pairs_list = []
                mapEstimate_list = []

        # Initialize lists to collect PDFs, or MAP estimates and HDI limits, or samples
        elif mode == 'data':
            probMassVec_list = []

        elif mode == 'summary':
            mapEstimate_list = []
            hdiLimits_list = []

        elif mode == 'samples':
            samples_list = []
            num_samples = 10000
            
        else:
            raise ValueError(f"Invalid value for mode: '{mode}'. Please use one of the following: 'plot', 'data', 'summary', 'samples'.")

        # Loop over sic values
        for k, sic_val in enumerate(sic):

            # Make predictions for all sets of parameters
            lnPIP_pred = (-np.log(1/sic_val-1) - b0) / b1

            # Compute PDFs for all sets of parameters, average them (along axis corresponding to different sets)
            pdfs = norm.pdf(lnPIP_grid[:, np.newaxis], lnPIP_pred, sd)
            avgdPDF = np.mean(pdfs, axis=1)

            # Normalise PDFs
            probMassVec = avgdPDF / np.sum(avgdPDF)
            # print('Sum of PDF:', np.sum(probMassVec))

            if mode == 'data':
                probMassVec_list.append(probMassVec)

            elif mode == 'samples':
                cdf = np.cumsum(probMassVec)
                cdf /= cdf[-1]   # normalize to 1
                random_vals = np.random.rand(num_samples)
                samples = np.interp(random_vals, cdf, lnPIP_grid)
                samples_list.append(samples)

            else:
                # Create lists of HDI limits (for single data point), find HDI
                lowerHDI_idx_list = []
                upperHDI_idx_list = []
                for m in hdiMass:
                    lowerHDI, upperHDI = self.HDIofGrid(probMassVec, m)
                    lowerHDI_idx_list.append(lowerHDI)
                    upperHDI_idx_list.append(upperHDI)

                # Find MAP estimate
                Imap = np.where(probMassVec == np.max(probMassVec))
                mapEstimate = lnPIP_grid[Imap]
                mapEstimate = mapEstimate[0]

                if mode == 'summary':
                    mapEstimate_list.append(mapEstimate)
                    hdiLimits = [(lnPIP_grid[lower], lnPIP_grid[upper]) for lower, upper in zip(lowerHDI_idx_list, upperHDI_idx_list)]
                    hdiLimits_list.append(hdiLimits)

                # Fill subplots / save results
                elif len(sic) == 1:
                        self.fill_subplots(axs, lowerHDI_idx_list, upperHDI_idx_list, lnPIP_grid, hdiMass,
                                           mapEstimate, probMassVec, c, c1)
                        self.forward_subplots(axs, sic_val, label)
                        axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

                elif len(sic) <= 6:
                    self.fill_subplots(axs[k], lowerHDI_idx_list, upperHDI_idx_list, lnPIP_grid, hdiMass,
                                       mapEstimate, probMassVec, c, c1)
                    self.forward_subplots(axs[k], sic_val, label)
                    if (len(sic) != 5 and k == len(sic)-1) or (len(sic) == 5 and k == 2):
                        axs[k].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

                else:
                    # Reverse list of lower HDI limits, extend with upper HDI limits
                    lowerHDI_idx_list = lowerHDI_idx_list[::-1]
                    idx_list = lowerHDI_idx_list + upperHDI_idx_list
                    idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

                    idx_pairs_list.append(idx_pairs)
                    mapEstimate_list.append(mapEstimate)

        # Plot series
        if mode == 'plot' and len(sic) > 6:
            fig, axs = plt.subplots(figsize=(0.5*len(sic)+1,3))
            x = np.arange(1, len(sic)+1)   # generate indices for x axis (+1 b/c end excluded)
            self.fill_series(axs, idx_pairs_list, lnPIP_grid, hdiMass, mapEstimate_list, x, c, c1)
            self.forward_series(axs, sic, x, label)

        if mode == 'plot':
            plt.tight_layout()
            return fig, axs

        elif mode == 'data':
            posteriors = np.vstack(probMassVec_list)   # stack into 2D array (len(sic), len(lnPIP_grid))
            df = pd.DataFrame(posteriors, columns=[lnPIP for lnPIP in lnPIP_grid])
            return df

        elif mode == 'summary':
            mapEstimation_array = np.array(mapEstimate_list)          # convert to array (len(sic),)
            hdiLimits_array = np.array(hdiLimits_list)                # (len(sic), len(hdiMass), 2)
            hdiLimits_array = hdiLimits_array.reshape(len(sic), -1)   # (len(sic), len(hdiMass)*2)
            
            column_names = ['MAPestimate']
            for m in hdiMass:
                column_names.extend([f'{m}HDI_lower', f'{m}HDI_upper'])
            df = pd.DataFrame(np.hstack((mapEstimation_array.reshape(-1, 1), hdiLimits_array)), columns=column_names)
            return df
        
        else:
            all_samples = np.vstack(samples_list)   # stack into 2D array (len(sic), num_samples)
            df = pd.DataFrame(all_samples, columns=[f'lnPIP25_{i+1}' for i in range(num_samples)])
            return df


    # Function to inverse model sic from lnPIP
    def inverse(self, ip25, sterol, index, unit, mode='plot', hdiMass=(0.15,0.35,0.55,0.75,0.95), xType='index', xVal=None):

        # Calculate lnPIP from ip25 and sterol
        lnPIP = self.cal_lnPIP(ip25, sterol, index, unit)

        # Ensure hdiMass is an array of floats and within valid range
        hdiMass = self.check_input(hdiMass, 'hdiMass')
        ip25 = self.check_input(ip25, 'IP₂₅')   # convert ip25 to array if not already (for length check)

        # Normalize input string to lowercase
        xType = str(xType).lower()
        
        # Ensure xVal and xType are both supplied, or both left as default
        if xVal is None:
            if xType != 'index':
                raise ValueError("Please supply xVal for each pair of biomarker measurements.")
        else:   # xVal is not None
            if xType == 'index':
                raise ValueError("Please specify xType: 'age', 'depth'.")
            # Check xType is either age or depth
            elif xType != 'age' and xType != 'depth':
                raise ValueError(f"Invalid value for xType: '{xType}'. Please use one of the following: 'age', 'depth'.")
            
            # Check xVal is an array of floats
            xVal = self.check_input(xVal, xType)
            # Ensure ip25 and xVal are of same length
            if len(xVal) != len(ip25):
                raise ValueError("The lengths of ip25 and xVal do not match. Please supply xVal for every pair of biomarker measurements.")

        # Format x-axis labels
        if xType == 'index':
            xVal = np.arange(1, len(ip25)+1)   # generate indices (+1 b/c end excluded)
            xlabel = 'Index'
        elif xType == 'age':
            xlabel = 'Age (ka BP)'
        elif xType == 'depth':
            xlabel = 'Depth (m)'

        # Select MAM posteriors, colours and labels
        if index == 'dino':
            filename = 'server_MAM_dino_7900_2024-08-05_05-03-41.npy'
            c = 'tab:red'
            c1 = 'darkred'
            label = r'$\ln(\mathrm{P_{D}IP_{25}})$'
        else:
            filename = 'server_MAM_bras_7900_2024-08-05_09-18-15.npy'
            c= 'tab:blue'
            c1 = 'darkblue'
            label = r'$\ln(\mathrm{P_{B}IP_{25}})$'

        # Construct path to file containing matrix calculated from corresponding posterior, load file
        filepath = os.path.join(self.current_dir, filename)
        avgdPDF = np.load(filepath)

        # Create evenly spaced grids within expected ranges of lnPIP and sic (should be the same as in matrix)
        lnPIP_grid = np.linspace(-12, 0, 10000)   # lnPIP lower limit...
        sic_grid = np.linspace(0, 1, 1000)
        sic_grid = sic_grid[1:-1]   # remove 0 and 1 to avoid division by 0 or log of 0

        # Find indices of 2 consecutive values between which lnPIP falls
        upper_idx = np.searchsorted(lnPIP_grid, lnPIP, side='right')
        lower_idx = upper_idx - 1

        # Get lnPIP_grid values
        upper_val = lnPIP_grid[upper_idx]
        lower_val = lnPIP_grid[lower_idx]

        # Get corresponding distributions (across sic_grid)
        upper_dist = avgdPDF[:,upper_idx]
        lower_dist = avgdPDF[:,lower_idx]

        # Calculate interpolation weight (0=lower value, 1=upper value)
        weight = (lnPIP-lower_val) / (upper_val-lower_val)

        # For up to 6 lnPIP values, create figure to plot PDF, HDI, and MAP estimate for each
        # Otherwise, create lists to save results for plotting later
        if mode == 'plot':
            if len(lnPIP) == 1:
                fig, axs = plt.subplots(figsize=(4.2,3))
            elif len(lnPIP) <= 6:
                fig, axs = self.create_subplots(len(lnPIP))
            else:
                idx_pairs_list = []
                mapEstimate_list = []

        # Initialize lists to collect PDFs, or MAP estimates and HDI limits
        elif mode == 'data':
            interpolatedPDF_list = []

        elif mode == 'summary':
            mapEstimate_list = []
            hdiLimits_list = []

        elif mode == 'samples':
            samples_list = []
            num_samples = 10000
            
        else:
            raise ValueError(f"Invalid value for mode: '{mode}'. Please use one of the following: 'plot', 'data', 'summary', 'samples'.")

        # Loop over lnPIP values
        for k, lnPIP_val in enumerate(lnPIP):

            # Normalize distributions to convert into PDFs
            upperPDF = upper_dist[:,k] / np.sum(upper_dist[:,k])
            lowerPDF = lower_dist[:,k] / np.sum(lower_dist[:,k])

            # Interpolate between the 2 PDFs
            interpolatedPDF = (1-weight[k])*lowerPDF + weight[k]*upperPDF
            # print('Sum of PDF:', np.sum(interpolatedPDF))

            if mode == 'data':
                interpolatedPDF_list.append(interpolatedPDF)

            elif mode == 'samples':
                cdf = np.cumsum(interpolatedPDF)
                cdf /= cdf[-1]   # normalize to 1
                random_vals = np.random.rand(num_samples)
                samples = np.interp(random_vals, cdf, sic_grid)
                samples_list.append(samples)

            else:
                # Create lists of HDI limits (for single data point), find HDI
                lowerHDI_idx_list = []
                upperHDI_idx_list = []
                for m in hdiMass:
                    lowerHDI, upperHDI = self.HDIofGrid(interpolatedPDF, m)
                    lowerHDI_idx_list.append(lowerHDI)
                    upperHDI_idx_list.append(upperHDI)

                # Find MAP estimate
                Imap = np.where(interpolatedPDF == np.max(interpolatedPDF))
                mapEstimate = sic_grid[Imap]
                mapEstimate = mapEstimate[0]

                if mode == 'summary':
                    mapEstimate_list.append(mapEstimate)
                    hdiLimits = [(sic_grid[lower], sic_grid[upper]) for lower, upper in zip(lowerHDI_idx_list, upperHDI_idx_list)]
                    hdiLimits_list.append(hdiLimits)

                # Fill subplot / save results
                elif len(lnPIP) == 1:
                    self.fill_subplots(axs, lowerHDI_idx_list, upperHDI_idx_list, sic_grid, hdiMass,
                                       mapEstimate, interpolatedPDF, c, c1)
                    self.inverse_subplots(axs, lnPIP_val, xVal[0], xlabel, label)
                    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

                elif len(lnPIP) <= 6:
                    self.fill_subplots(axs[k], lowerHDI_idx_list, upperHDI_idx_list, sic_grid, hdiMass,
                                       mapEstimate, interpolatedPDF, c, c1)
                    self.inverse_subplots(axs[k], lnPIP_val, xVal[k], xlabel, label)
                    if (len(lnPIP) != 5 and k == len(lnPIP)-1) or (len(lnPIP) == 5 and k == 2):
                        axs[k].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

                else:
                    # Reverse list of lower HDI limits, extend with upper HDI limits
                    lowerHDI_idx_list = lowerHDI_idx_list[::-1]
                    idx_list = lowerHDI_idx_list + upperHDI_idx_list
                    idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

                    idx_pairs_list.append(idx_pairs)
                    mapEstimate_list.append(mapEstimate)

        # Plot series
        if mode == 'plot' and len(lnPIP) > 6:
            fig, axs = plt.subplots(figsize=(0.5*len(lnPIP)+1,3))
            self.fill_series(axs, idx_pairs_list, sic_grid, hdiMass, mapEstimate_list, xVal, c, c1)
            self.inverse_series(axs, xVal, xlabel, label)

        if mode == 'plot':
            plt.tight_layout()
            return fig, axs
        
        elif mode == 'data':
            posteriors = np.vstack(interpolatedPDF_list)   # stack into 2D array (len(lnPIP), len(sic_grid))
            df = pd.DataFrame(posteriors, columns=[sic for sic in sic_grid])
            return df
        
        elif mode == 'summary':
            mapEstimation_array = np.array(mapEstimate_list)            # convert to array (len(lnPIP),)
            hdiLimits_array = np.array(hdiLimits_list)                  # (len(lnPIP), len(hdiMass), 2)
            hdiLimits_array = hdiLimits_array.reshape(len(lnPIP), -1)   # (len(lnPIP), len(hdiMass)*2)
            
            column_names = ['MAPestimate']
            for m in hdiMass:
                column_names.extend([f'{m}HDI_lower', f'{m}HDI_upper'])
            df = pd.DataFrame(np.hstack((mapEstimation_array.reshape(-1, 1), hdiLimits_array)), columns=column_names)
            return df
        
        else:
            all_samples = np.vstack(samples_list)   # stack into 2D array (len(lnPIP), num_samples)
            df = pd.DataFrame(all_samples, columns=[f'SIC_{i+1}' for i in range(num_samples)])
            return df


    # Function to determine the month of the first SIC decrease given a 1D array of monthly SIC climatology
    def find_first_decrease(self, sic_climo):

        # Check if sea ice occurs (i.e., annual mean SIC > 0)
        annual_sic = np.nanmean(sic_climo)
        if np.isnan(annual_sic) or annual_sic == 0:
            raise ValueError("SIC must not be all NaN values or all zeros. Please input a [12 x lat x lon] spatial field of climatologies instead.")

        # Round to the nearest 5%
        sic_rounded = np.round(sic_climo / 0.05) * 0.05
        
        # Stack the climatology twice for wrapping around the year
        sic_stacked = np.concatenate([sic_rounded, sic_rounded])

        # Find the indices of the maximum value(s)
        Imax = np.where(sic_rounded == np.max(sic_rounded))[0]

        # Check that not all SIC values are the same
        if len(Imax) == 12:
            raise ValueError("SIC is constant throughout the year. Please input a [12 x lat x lon] spatial field of climatologies instead.")
        
        # If only one maximum value exists, just return that value
        elif len(Imax) == 1:
            first_decrease = Imax[0]

        # If multiple maximum values exist, return the value with the lowest corresponding minimum
        else:
            min_vals = np.full(len(Imax), np.nan)   # preallocate (for sanity checking)
            Imin = np.full(len(Imax), np.nan)       # preallocate
            
            for k in range(len(Imax)):   # for each maximum value...
                m = 1                    # create a roaming index to save all minimum values and month indices

                # If the next month contains the same SIC as the current month (i.e., the maximum value), move on to
                # the next maximum value; otherwise, loop through subsequent months until a local minimum value is found
                next_sic = sic_stacked[Imax[k] + m]
                if next_sic != sic_stacked[Imax[k]]: 
                    while next_sic <= sic_stacked[Imax[k] + m - 1]:
                        m += 1   # update roaming index
                        next_sic = sic_stacked[Imax[k] + m]
                
                # Save the minimum values and corresponding month indices
                min_vals[k] = sic_stacked[Imax[k] + m - 1]
                Imin[k] = Imax[k] + m - 1

            # Convert Imin and Imax to 0-11
            Imax[Imax > 11] -= 12
            Imin[Imin > 11] -= 12

            # Convert Imin and Imax to integer type to be used as indices
            Imin = Imin.astype(int)
            Imax = Imax.astype(int)
            
            # Find the index corresponding to the largest decrease from maximum to minimum
            min_diff_idx = np.argmin(sic_stacked[Imin] - sic_stacked[Imax])
            first_decrease = Imax[min_diff_idx]

        return first_decrease


    # Function to calculate the mean SIC of the 3 months before the first SIC decrease
    # sic_climo: [12 x 1] monthly SIC climatology
    # first_decrease (optional): integer (0-11), month of the first SIC decrease
    def cal_meanSIC_1D(self, sic_climo, first_decrease=None):

        # Find the month of the first SIC decrease if not provided
        if first_decrease is None:
            first_decrease = self.find_first_decrease(sic_climo)

        # Get the first decrease month and the two previous months
        months_used = [(first_decrease - i) % 12 for i in range(3)]
        months_used = months_used[::-1]                      # reverse the order to be chronological
        sic_used = [sic_climo[idx] for idx in months_used]   # extract SIC of selected months
        meanSIC = np.nanmean(sic_used)                       # calculate mean SIC

        return meanSIC, months_used


    # Function to calculate Euclidean distances between a locality and
    # all combinations of latitudes and longitudes using Haversine formula
    # site_lat, site_lon: scalars (in radians)
    # all_lat, all_lon: 2D arrays (in radians)
    def haversine_vectorized(self, site_lat, site_lon, all_lat, all_lon):

        dlat = all_lat - site_lat
        dlon = all_lon - site_lon
        a = np.sin(dlat/2)**2 + np.cos(site_lat) * np.cos(all_lat) * np.sin(dlon/2)**2
        a[a > 1] = 1                                     # ensure a does not exceed 1 due to rounding errors
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))   # central angle in radians
        R = 6371                                         # Earth's radius in km
        distances = R * c

        return distances


    # Function to find the nearest grid cell with a non-NaN value
    # values: [n x lat x lon]
    # site_lat, site_lon: scalars (in radians)
    # all_lat, all_lon: 2D arrays (in radians)
    def find_nearest_non_nan(self, values, site_lat, site_lon, all_lat, all_lon):
        
        # Flatten values to 2D array [n x (lat*lon)]
        values_flattened = values.reshape(values.shape[0], -1)
        
        # Calculate Euclidean distances between locality and all model grid points
        distances = self.haversine_vectorized(site_lat, site_lon, all_lat, all_lon)
        # Sort distances in ascending order and get sorting indices
        Isort = np.argsort(distances, axis=None)

        # Flatten 2D matrix, sort with Isort
        distances_sorted = distances.flatten()[Isort]
        values_sorted = values_flattened[:, Isort]   # also sort flattened values

        # Go through (first) values from the closest grid to the furthest
        # If NaN occurs, move to next closest
        num_nan = 0
        while np.isnan(values_sorted[0, num_nan]):
            num_nan += 1

        nearest_non_nan = values_sorted[:, num_nan]    # value from the nearest non-NaN grid cell
        matched_distance = distances_sorted[num_nan]   # distance between locality and matched grid

        # Find row and column indices corresponding to num_nan
        row_idx, col_idx = np.unravel_index(Isort[num_nan], distances.shape)

        # Get corresponding latitude and longitude from all_lat and all_lon, convert to degrees
        matched_lat = np.degrees(all_lat[row_idx, col_idx])
        matched_lon = np.degrees(all_lon[row_idx, col_idx])

        return nearest_non_nan, num_nan, matched_distance, matched_lat, matched_lon


    # User-facing function
    # sic_climo: [12 x 1] OR [12 x lat x lon] monthly SIC climatology
    # site_lat, site_lon: scalars (in degrees), latitude and longitude of the target site
    # all_lat, all_lon: 1D arrays (in degrees), latitudes and longitudes corresponding to sic_climo
    def cal_meanSIC(self, sic_climo, site_lat=None, site_lon=None, all_lat=None, all_lon=None):

        # Convert to numpy array if not already
        sic_climo = np.array(sic_climo)

        # Check if SIC values are between 0 and 1
        if (sic_climo<0).any() or (sic_climo>1).any():
            raise ValueError("All SIC values must be between 0 and 1.")
        
        input_shape = sic_climo.shape

        # For 1D input, calculate mean SIC
        # cal_meanSIC_1D() raises error if sea ice does not occur or SIC is constant throughout the year
        if input_shape == (12,) or input_shape == (12, 1) or input_shape == (1, 12):
            sic_climo = np.squeeze(sic_climo)   # convert to 1D if not already
            results = self.cal_meanSIC_1D(sic_climo)
            return results

        # For 3D input, determine first decrease month for all grid cells
        # Get the first decrease month from the nearest grid cell where sea ice occurs + varies
        # Calculate mean SIC from the nearest non-NaN grid cell using that month
        elif len(input_shape) == 3 and input_shape[0] == 12:

            # Check if site_lat, site_lon, all_lat, all_lon are provided
            if site_lat is None or site_lon is None or all_lat is None or all_lon is None:
                raise ValueError("For 3D SIC climatology input, please also provide site_lat, site_lon, all_lat, and all_lon.")

            # Change all invalid latitudes and longitudes to NaN
            all_lat[(all_lat < -90) | (all_lat > 90)] = np.nan
            all_lon[(all_lon < -180) | (all_lon > 360)] = np.nan

            # Normalize longitudes to -180 to 180
            site_lon = (site_lon + 180) % 360 - 180
            all_lon = (all_lon + 180) % 360 - 180

            # Convert degrees to radians
            site_lat = np.radians(site_lat)
            site_lon = np.radians(site_lon)
            all_lat = np.radians(all_lat)
            all_lon = np.radians(all_lon)

            # Convert latitudes and longitudes to 2D arrays if not already (for broadcasting)
            if all_lat.ndim == 1 and all_lon.ndim == 1:
                all_lon, all_lat = np.meshgrid(all_lon, all_lat)
            elif all_lon.ndim != 2 or all_lat.ndim != 2 or all_lon.shape != all_lat.shape:
                raise ValueError("all_lat and all_lon must be 1D arrays, or 2D arrays of the same shape.")

            # Check if all_lat and all_lon match the spatial dimensions of sic_climo
            if all_lon.shape != input_shape[1:]:
                raise ValueError(f"The dimensions of all_lat and all_lon {all_lon.shape} do not match the spatial dimensions of sic_climo {input_shape[1:]}.")

            # Find the first decrease month for all grid cells
            first_decrease_all = np.full(input_shape[1:], np.nan)   # preallocate
            for i in range(input_shape[1]):
                for j in range(input_shape[2]):
                    try:
                        first_decrease_all[i, j] = self.find_first_decrease(sic_climo[:, i, j])
                    except ValueError:   # if sea ice does not occur or SIC is constant,
                        continue         # leave as NaN and move on to the next grid cell

            # Expand dims to [1 x lat x lon] for find_nearest_non_nan function, get first decrease month
            first_decrease_all = np.expand_dims(first_decrease_all, axis=0)
            matched_first_decrease = self.find_nearest_non_nan(first_decrease_all, site_lat, site_lon, all_lat, all_lon)
            first_decrease = int(matched_first_decrease[0])   # extract first decrease month as integer

            # Get SIC climatology, calculate mean SIC using determined month
            matched_sic = self.find_nearest_non_nan(sic_climo, site_lat, site_lon, all_lat, all_lon)
            matched_sic_climo = matched_sic[0]   # extract SIC climatology
            results = self.cal_meanSIC_1D(matched_sic_climo, first_decrease)
            return results

        else:
            raise ValueError("SIC climatology must be a 1D array of length 12, or a 3D array with the 1st dimension of length 12.")


    def cat(self):

        fig, ax = plt.subplots(figsize=(4,3))
        ax.axis('off')
        ax.text(0.5, 0.5, " ∧,,,∧   ♪     \n (• ˕ •)          \n---- U  U ------------\n| ʜᴀᴠᴇ ᴀ ɴɪᴄᴇ ᴅᴀʏ! |\n------------------------",
                fontsize=20, ha='center', va='center')
        # ax.text(0.5, 0.5,
        #         " ∧,,,∧         \n (ᐠ ˕ ᐟ)         \n------ U  U -------------\n| ɴᴏᴛ ᴜ ᴀɢᴀɪɴ! (ᴊᴋ:ᴘ) |\n---------------------------",
        #         fontsize=20, ha='center', va='center')
        
        return fig
