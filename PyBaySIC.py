# BaySIC! :D

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Required input:
# 1. predictor
#       forward -> sic (0-1)
#       inverse -> ip25 and sterol (>=0), in same units!!
# 2. index ('dino'/'bras')
# 3. unit ('toc'/'sed'), for inverse model only

# Optional input:
# 4. hdiMass (0-1), default to (0.15, 0.35, 0.55, 0.75, 0.95)
# 5a. xType ('age'/'depth'), for inverse model only, default to index
# 5b. xVal (>=0, in ascending/descending order), for inverse model only
#       age expected in ka BP, depth expected in m

# Calibration interval
# forward -> 3 months before 1st SIC decrease
# inverse -> MAM

# Output:
# kernel distribution (for up to 6 predictions)
# highest density interval (HDI)
# maximum a posteriori (MAP) estimation

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# These values are added to all biomarker inputs in the inverse function
min_ip25_toc = 0.01049169444307899
min_ip25_sed = 6.212970072165696e-05
min_dino_toc = 0.21311343280837622
min_dino_sed = 0.0008602308083664112
min_bras_toc = 0.14608475206034371
min_bras_sed = 0.0005896700301752964


try:
    import os
except ImportError:
    print ("Please install os.")

try:
    import numpy as np
except ImportError:
    print ("Please install NumPy.")

try:
    from tqdm import tqdm
except ImportError:
    print ("Please install tqdm.")

try:
    from scipy.stats import norm
except ImportError:
    print ("Please install SciPy.")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print ("Please install Matplotlib.")


class BaySIC:

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


    # Function for plotting forward modelling results (lnPIP from sic) in subplots
    def forward_subplots(self, ax, lnPIP_grid, probMassVec, lowerHDI_idx_list, upperHDI_idx_list,
                         mapEstimation, sic_val, hdiMass, c, c1, label):
        
        # Reverse list of lower HDI limits, extend with upper HDI limits, create pairs
        # E.g. hdiMass = 0.3,0.6,0.9, lower limits (indices) = 30,20,10, upper limits = 40,50,60
        # idx_list = 0,10,20,30,40,50,60,70, idx_pairs = [0,10],[10,20],[20,30],...,[60,70]
        lowerHDI_idx_list = lowerHDI_idx_list[::-1]
        idx_list = [0] + lowerHDI_idx_list + upperHDI_idx_list + [len(lnPIP_grid)-1]
        idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

        # Create array of alphas, mirror itself (-2 = start from 2nd last element, ::-1 = reverse array)
        alphas = np.insert(hdiMass, 0, 0.05)
        alphas_m = np.concatenate((alphas, alphas[-2::-1]))
        alphas_m = alphas_m*0.9

        # Reverse hdiMass, add '100% HDI', mirror itself
        hdiMass_r = hdiMass[::-1]
        hdiMass_r = np.insert(hdiMass_r, 0, 1)
        hdiMass_m = np.concatenate((hdiMass_r, hdiMass_r[-2::-1]))

        # Plot MAP estimation as vertical line, add label
        ax.axvline(x=mapEstimation, color=c1, alpha=0.75, label='MAP estimation')
        ax.text(mapEstimation-0.15, max(probMassVec)*1.05, f'{round(mapEstimation,2)}', ha='right', va='center')

        # Shade HDI
        k=0
        for pair, alpha, m in zip(idx_pairs, alphas_m, hdiMass_m):

            x = lnPIP_grid[pair[0]: pair[1]]
            y = probMassVec[pair[0]: pair[1]]

            if k < int(len(hdiMass_m)/2):
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None)
            elif k == len(hdiMass_m)-1:
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None, label=f'>{round(hdiMass_m[-2]*100)}% HDI')
            else:
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None, label=f'{round(m*100)}% HDI')

            k+=1

        ax.set_xlim(left=-12, right=0)   # lnPIP lower limit...
        ax.set_ylim(bottom=0, top=max(probMassVec)*1.1)

        ax.set_xlabel(label)
        ax.set_ylabel('P')
        ax.yaxis.set_ticks([])
        ax.set_title(f'SIC = {round(sic_val, 3)}')   # if input SIC = 0/1, this shows treated SIC


    # Function for plotting inverse modelling results (sic from lnPIP) in subplots
    def inverse_subplots(self, ax, sic_grid, interpolatedPDF, lowerHDI_idx_list, upperHDI_idx_list,
                         mapEstimation, lnPIP_val, hdiMass, xVal, xlabel, c, c1, label):

        # Reverse list of lower HDI limits, extend with upper HDI limits, create pairs
        # E.g. hdiMass = 0.3,0.6,0.9, lower limits (indices) = 30,20,10, upper limits = 40,50,60
        # idx_list = 0,10,20,30,40,50,60,70, idx_pairs = [0,10],[10,20],[20,30],...,[60,70]
        lowerHDI_idx_list = lowerHDI_idx_list[::-1]
        idx_list = [0] + lowerHDI_idx_list + upperHDI_idx_list + [len(sic_grid)-1]
        idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

        # Create array of alphas, mirror itself (-2 = start from 2nd last element, ::-1 = reverse array)
        alphas = np.insert(hdiMass, 0, 0.05)
        alphas_m = np.concatenate((alphas, alphas[-2::-1]))
        alphas_m = alphas_m*0.9

        # Reverse hdiMass, add '100% HDI', mirror itself
        hdiMass_r = hdiMass[::-1]
        hdiMass_r = np.insert(hdiMass_r, 0, 1)
        hdiMass_m = np.concatenate((hdiMass_r, hdiMass_r[-2::-1]))
        
        # Plot MAP estimation as vertical line, add label
        ax.axvline(x=mapEstimation, color=c1, alpha=0.75, label='MAP estimation')
        ax.text(mapEstimation-0.01, max(interpolatedPDF)*1.05, f'{round(mapEstimation,2)}', ha='right', va='center')

        # Shade HDI
        k=0
        for pair, alpha, m in zip(idx_pairs, alphas_m, hdiMass_m):

            x = sic_grid[pair[0]: pair[1]]
            y = interpolatedPDF[pair[0]: pair[1]]

            if k < int(len(hdiMass_m)/2):
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None)
            elif k == len(hdiMass_m)-1:
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None, label=f'>{round(hdiMass_m[-2]*100)}% HDI')
            else:
                ax.fill_between(x, y, color=c, alpha=alpha, edgecolor=None, label=f'{round(m*100)}% HDI')
            
            k+=1

        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=max(interpolatedPDF)*1.1)

        ax.set_xlabel('SIC')
        ax.set_ylabel('P')
        ax.yaxis.set_ticks([])

        if xlabel == 'Age (ka BP)':
            ax.set_title(f'{xVal} ka BP, {label} = {round(lnPIP_val, 3)}')
        elif xlabel == 'Depth (m)':
            ax.set_title(f'{xVal} m, {label} = {round(lnPIP_val, 3)}')
        else:   # index not needed
            ax.set_title(f'{label} = {round(lnPIP_val, 3)}')


    # Function for plotting forward modelling results (lnPIP from sic) as series
    def forward_series(self, ax, idx_pairs_list, lnPIP_grid, mapEstimation_list, sic, hdiMass, c, c1, label):

        # Generate indices for x axis (+1 b/c end excluded)
        x = np.arange(1, len(sic)+1)

        # Create array of alphas, mirror itself (-2 = start from 2nd last element, ::-1 = reverse array)
        alphas_m = np.concatenate((hdiMass, hdiMass[-2::-1]))
        alphas_m = alphas_m*0.8

        # Reverse hdiMass, mirror itself
        hdiMass_r = hdiMass[::-1]
        hdiMass_m = np.concatenate((hdiMass_r, hdiMass_r[-2::-1]))

        # Convert list to array, switch axis 0 and 1 (sic index, HDI)
        idx_pairs_array = np.array(idx_pairs_list)
        transposed_array = np.transpose(idx_pairs_array, (1, 0, 2))

        # Plot MAP estimations as broken line
        ax.plot(x, mapEstimation_list, color=c1, clip_on=False, label='MAP estimation')

        # Iterate over axis 0 (HDI), shade HDI
        for i, sub_array in enumerate(transposed_array):

            y1 = lnPIP_grid[sub_array[:,0]]
            y2 = lnPIP_grid[sub_array[:,1]]

            if i < int(len(hdiMass_m)/2):
                ax.fill_between(x, y1, y2, color=c, alpha=alphas_m[i], edgecolor=None)
            else:
                ax.fill_between(x, y1, y2, color=c, alpha=alphas_m[i], edgecolor=None,
                                label=f'{round(hdiMass_m[i]*100)}% HDI')
        
        ax.set_xlim(left=1, right=len(sic))
        ax.set_ylim(bottom=-12, top=0)   # lnPIP lower limit...

        ax.set_xticks(x)
        ax.set_xlabel('Index')
        ax.set_ylabel(label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)


    # Function for plotting inverse modelling results (sic from lnPIP) as series
    def inverse_series(self, ax, idx_pairs_list, sic_grid, mapEstimation_list, hdiMass, xVal, xlabel, c, c1, label):

        # Create array of alphas, mirror itself (-2 = start from 2nd last element, ::-1 = reverse array)
        alphas_m = np.concatenate((hdiMass, hdiMass[-2::-1]))
        alphas_m = alphas_m*0.8

        # Reverse hdiMass, mirror itself
        hdiMass_r = hdiMass[::-1]
        hdiMass_m = np.concatenate((hdiMass_r, hdiMass_r[-2::-1]))

        # Convert list to array, switch axis 0 and 1 (lnPIP index, HDI)
        idx_pairs_array = np.array(idx_pairs_list)
        transposed_array = np.transpose(idx_pairs_array, (1, 0, 2))

        # Plot MAP estimations as broken line
        ax.plot(xVal, mapEstimation_list, color=c1, clip_on=False, label='MAP estimation')

        # Iterate over axis 0 (HDI), shade HDI
        for i, sub_array in enumerate(transposed_array):

            y1 = sic_grid[sub_array[:,0]]
            y2 = sic_grid[sub_array[:,1]]

            if i < int(len(hdiMass_m)/2):
                ax.fill_between(xVal, y1, y2, color=c, alpha=alphas_m[i], edgecolor=None)
            else:
                ax.fill_between(xVal, y1, y2, color=c, alpha=alphas_m[i], edgecolor=None,
                                label=f'{round(hdiMass_m[i]*100)}% HDI')

        ax.set_xlim(left=np.min(xVal), right=np.max(xVal))
        ax.set_ylim(bottom=0, top=1)

        if xlabel == 'Index':
            ax.set_xticks(xVal.astype(int))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('SIC')

        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        legend.set_title(f'Predictor: {label}')
        legend._legend_box.align = 'left'



    # Function for forward modelling lnPIP from sic
    def forward(self, sic, index, hdiMass=(0.15,0.35,0.55,0.75,0.95)):

        # Ensure sic and hdiMass are arrays of floats and within valid ranges
        sic = self.check_input(sic, 'SIC')
        hdiMass = self.check_input(hdiMass, 'hdiMass')

        # Normalize input string to lowercase
        index = str(index).lower()

        # Select '3 months before 1st sic loss' posteriors, colours and labels
        if index == 'dino':
            filename = 'server_dino_spavar_7900_2024-09-06_08-56-41.txt'
            c = 'r'
            c1 = 'darkred'
            label = r'$\ln(\mathrm{P_{D}IP_{25}})$'
        elif index == 'bras':
            filename = 'server_bras_spavar_7900_2024-09-06_08-56-41.txt'
            c= 'royalblue'
            c1 = 'navy'
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

        # For up to 6 sic values, create figure to plot PDF, HDI, and MAP estimation for each
        if len(sic)== 1:
            fig, axs = plt.subplots(figsize=(4.2,3))
        elif len(sic) <= 6:
            fig, axs = self.create_subplots(len(sic))

        # Otherwise, create lists to save results for plotting later
        else:
            mapEstimation_list = []
            idx_pairs_list = []

        # Loop over sic values
        for k, sic_val in tqdm(enumerate(sic), desc='Processing'):

            # Make predictions for all sets of parameters
            lnPIP_pred = (-np.log(1/sic_val-1) - b0) / b1

            # Compute PDFs for all sets of parameters, average them (along axis corresponding to different sets)
            pdfs = norm.pdf(lnPIP_grid[:, np.newaxis], lnPIP_pred, sd)
            avgdPDF = np.mean(pdfs, axis=1)

            # Normalise PDFs
            probMassVec = avgdPDF / np.sum(avgdPDF)
            # print('Sum of PDF:', np.sum(probMassVec))

            # Create lists of HDI limits (for single data point), find HDI
            lowerHDI_idx_list = []
            upperHDI_idx_list = []
            for m in hdiMass:
                lowerHDI, upperHDI = self.HDIofGrid(probMassVec, m)
                lowerHDI_idx_list.append(lowerHDI)
                upperHDI_idx_list.append(upperHDI)

            # Find MAP estimation
            Imap = np.where(probMassVec == np.max(probMassVec))
            mapEstimation = lnPIP_grid[Imap]
            mapEstimation = mapEstimation[0]
            print(mapEstimation)

            # Fill subplot / save results
            if len(sic) == 1:
                self.forward_subplots(axs, lnPIP_grid, probMassVec, lowerHDI_idx_list, upperHDI_idx_list,
                                      mapEstimation, sic_val, hdiMass, c, c1, label)
                axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

            elif len(sic) <= 6:
                self.forward_subplots(axs[k], lnPIP_grid, probMassVec, lowerHDI_idx_list, upperHDI_idx_list,
                                      mapEstimation, sic_val, hdiMass, c, c1, label)
                if (len(sic) != 5 and k == len(sic)-1) or (len(sic) == 5 and k == 2):
                    axs[k].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

            else:
                # Reverse list of lower HDI limits, extend with upper HDI limits
                lowerHDI_idx_list = lowerHDI_idx_list[::-1]
                idx_list = lowerHDI_idx_list + upperHDI_idx_list
                idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

                idx_pairs_list.append(idx_pairs)
                mapEstimation_list.append(mapEstimation)

        # Plot series
        if len(sic) > 6:
            fig, axs = plt.subplots(figsize=(0.5*len(sic)+1,3))
            self.forward_series(axs, idx_pairs_list, lnPIP_grid, mapEstimation_list, sic, hdiMass, c, c1, label)

        plt.tight_layout()

        return fig, axs
    


    # Function for inverse modelling sic from lnPIP
    def inverse(self, ip25, sterol, index, unit, hdiMass=(0.15,0.35,0.55,0.75,0.95), xType='index', xVal=None):
        
        # Ensure ip25, sterol, and hdiMass are arrays of floats and within valid ranges
        ip25 = self.check_input(ip25, 'IP₂₅')
        sterol = self.check_input(sterol, 'sterol')
        hdiMass = self.check_input(hdiMass, 'hdiMass')

        # Ensure ip25 and sterol are of same length
        if len(ip25) != len(sterol):
            raise ValueError("The lengths of ip25 and sterol do not match. Please use paired measurements.")
        
        # Normalize input strings to lowercase
        index = str(index).lower()
        unit = str(unit).lower()
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
            c = 'r'
            c1 = 'darkred'
            label = r'$\ln(\mathrm{P_{D}IP_{25}})$'
        elif index == 'bras':
            filename = 'server_MAM_bras_7900_2024-08-05_09-18-15.npy'
            c= 'royalblue'
            c1 = 'navy'
            label = r'$\ln(\mathrm{P_{B}IP_{25}})$'
        else:
            raise ValueError(f"Invalid value for index: '{index}'. Please use one of the following: 'dino', 'bras'.")

        # Treat biomarker concentrations
        if unit == 'toc':
            ip25 += min_ip25_toc
            if index == 'dino':
                sterol += min_dino_toc
            else:
                sterol += min_bras_toc
        elif unit == 'sed':
            ip25 += min_ip25_sed
            if index == 'dino':
                sterol += min_dino_sed
            else:
                sterol += min_bras_sed
        else:
            raise ValueError(f"Invalid value for unit: '{unit}'. Please use one of the following: 'toc', 'sed'.")

        # Calculate lnPIP
        lnPIP = np.log(ip25/(ip25+sterol))

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

        # For up to 6 lnPIP values, create figure to plot PDF, HDI, and MAP estimation for each
        if len(lnPIP)== 1:
            fig, axs = plt.subplots(figsize=(4.2,3))
        elif len(lnPIP) <= 6:
            fig, axs = self.create_subplots(len(lnPIP))

        # Otherwise, create lists to save results for plotting later
        else:
            mapEstimation_list = []
            idx_pairs_list = []
        
        # Loop over lnPIP values
        for i, lnPIP_val in tqdm(enumerate(lnPIP), desc='Processing'):

            # Normalize distributions to convert into PDFs
            upperPDF = upper_dist[:,i] / np.sum(upper_dist[:,i])
            lowerPDF = lower_dist[:,i] / np.sum(lower_dist[:,i])

            # Interpolate between the 2 PDFs
            interpolatedPDF = (1-weight[i])*lowerPDF + weight[i]*upperPDF
            # print('Sum of PDF:', np.sum(interpolatedPDF))

            # Create lists of HDI limits (for single data point), find HDI
            lowerHDI_idx_list = []
            upperHDI_idx_list = []
            for m in hdiMass:
                lowerHDI, upperHDI = self.HDIofGrid(interpolatedPDF, m)
                lowerHDI_idx_list.append(lowerHDI)
                upperHDI_idx_list.append(upperHDI)

            # Find MAP estimation
            Imap = np.where(interpolatedPDF == np.max(interpolatedPDF))
            mapEstimation = sic_grid[Imap]
            mapEstimation = mapEstimation[0]

            # Fill subplot / save results
            if len(lnPIP) == 1:
                self.inverse_subplots(axs, sic_grid, interpolatedPDF, lowerHDI_idx_list, upperHDI_idx_list,
                                      mapEstimation, lnPIP_val, hdiMass, xVal[0], xlabel, c, c1, label)
                axs.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

            elif len(lnPIP) <= 6:
                self.inverse_subplots(axs[i], sic_grid, interpolatedPDF, lowerHDI_idx_list, upperHDI_idx_list,
                                      mapEstimation, lnPIP_val, hdiMass, xVal[i], xlabel, c, c1, label)
                if (len(lnPIP) != 5 and i == len(lnPIP)-1) or (len(lnPIP) == 5 and i == 2):
                    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

            else:
                # Reverse list of lower HDI limits, extend with upper HDI limits
                lowerHDI_idx_list = lowerHDI_idx_list[::-1]
                idx_list = lowerHDI_idx_list + upperHDI_idx_list
                idx_pairs = [[idx_list[i], idx_list[i+1]] for i in range(len(idx_list)-1)]

                idx_pairs_list.append(idx_pairs)
                mapEstimation_list.append(mapEstimation)

        # Plot series
        if len(lnPIP) > 6:
            fig, axs = plt.subplots(figsize=(0.5*len(lnPIP)+1,3))
            self.inverse_series(axs, idx_pairs_list, sic_grid, mapEstimation_list, hdiMass, xVal, xlabel, c, c1, label)

        plt.tight_layout()

        return fig, axs
    


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
            ip25 += min_ip25_toc
            if index == 'dino':
                sterol += min_dino_toc
            else:
                sterol += min_bras_toc
        elif unit == 'sed':
            ip25 += min_ip25_sed
            if index == 'dino':
                sterol += min_dino_sed
            else:
                sterol += min_bras_sed
        else:
            raise ValueError(f"Invalid value for unit: '{unit}'. Please use one of the following: 'toc', 'sed'.")

        # Calculate lnPIP
        lnPIP = np.log(ip25/(ip25+sterol))

        return lnPIP
    


    def cat(self):

        fig, ax = plt.subplots(figsize=(4,3))
        ax.axis('off')
        ax.text(0.5, 0.5, " ∧,,,∧   ♪     \n (• ˕ •)          \n---- U  U ------------\n| ʜᴀᴠᴇ ᴀ ɴɪᴄᴇ ᴅᴀʏ! |\n------------------------",
                fontsize=20, ha='center', va='center')
        # ax.text(0.5, 0.5,
        #         " ∧,,,∧         \n (ᐠ ˕ ᐟ)         \n------ U  U -------------\n| ɴᴏᴛ ᴜ ᴀɢᴀɪɴ! (ᴊᴋ:ᴘ) |\n---------------------------",
        #         fontsize=20, ha='center', va='center')
        
        return fig