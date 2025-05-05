class BarFits:
    def __init__(self, data, *, conc_col='concentration', fracmort_col='mortality', 
                 btProtein_col='btProtein', population_col='population', 
                 replicate_col='replicate'):
        self.df = data
        self.conc_col = conc_col
        self.fracmort_col = fracmort_col
        self.btProtein_col = btProtein_col
        self.population_col = population_col
        self.replicate_col = replicate_col
        self.bioassay = self.df[self.btProtein_col].unique().tolist()
        self.populations = {
            btProtein: self.df[self.df[self.btProtein_col] == btProtein][self.population_col].unique().tolist() 
            for btProtein in self.bioassay
        }

    def plotPop(self, bioassay='all', populations='all', max_populations_per_subplot=5, ncol=3, all_subplots='sFAW', ylabel='Mortality', xlabel='Concentration'):
        """Plot bar graphs for selected bioassays and populations."""
        
        # Select bioassays
        if bioassay == 'all':
            selected_bioassays = self.bioassay
        else:
            selected_bioassays = bioassay

        # Select populations
        if populations == 'all':
            selected_populations = list(collections.OrderedDict.fromkeys([pop for btProtein in selected_bioassays for pop in self.populations[btProtein]]))
        else:
            selected_populations = populations
        
        # Ensure all_subplots is included in the selected populations
        if all_subplots and all_subplots not in selected_populations:
            selected_populations.append(all_subplots)

        # Set up the number of rows needed
        num_rows = (len(selected_bioassays) + ncol - 1) // ncol  # Ceiling division

        # Set up the figure and axes for subplots
        fig, axes = plt.subplots(nrows=num_rows, ncols=ncol, figsize=(15, num_rows * 4), sharex=True)
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D

        # Define a color palette
        CBPALETTE = ('#999999', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

        # Loop through each selected bioassay and create a subplot
        for i, btProtein in enumerate(selected_bioassays):
            subset = self.df[self.df[self.btProtein_col] == btProtein]

            # Prepare to plot the populations
            bar_data = []
            for population in selected_populations:
                pop_data = subset[subset[self.population_col] == population]
                if not pop_data.empty:
                    bar_data.append((pop_data[self.conc_col].values, pop_data[self.fracmort_col].values, pop_data['SEM'].values))

            # Ensure there is data to plot
            if not bar_data:
                print(f'No data available for bioassay: {btProtein} with populations: {selected_populations}')
                continue

            # Set bar width
            bar_width = 0.6
            # Set bar width based on the number of populations
            #total_populations = len(selected_populations)
            #bar_width = max(0.4, 0.8 / total_populations)  # Ensure the width does not exceed 0.8


            # Plotting the bar plot for each population with offsets
            for j, (conc, mortality, sem) in enumerate(bar_data[:max_populations_per_subplot]):
                # Calculate the x position for each population
                x_positions = np.arange(len(conc)) + (j * bar_width)  # Offset each population by bar_width
                axes[i].bar(x_positions, mortality, yerr=sem, 
                            capsize=5,  
                            lw = 3,
                            color=CBPALETTE[j % len(CBPALETTE)],
                            alpha=0.7,
                            width=bar_width,
                            edgecolor= 'black',
                            label=selected_populations[j])

            # Adding labels and title for each subplot
            axes[i].set_title(f'{btProtein}', fontsize=14)
            axes[i].set_ylabel(ylabel, fontsize=16, fontweight='bold', fontname='Arial')
            axes[i].tick_params(axis='both', which='major', labelsize=12)
            axes[i].grid(True, linestyle='', alpha=0.7)  # Styled grid

            # Set x-ticks to the unique concentration values
            unique_conc = np.unique(np.concatenate([bar_data[k][0] for k in range(len(bar_data))]))
            axes[i].set_xticks(np.arange(len(unique_conc)))  # Set positions for x-ticks
            axes[i].set_xticklabels(unique_conc)  # Set the labels to the unique concentrations

            # Set y-limits to ensure visibility of bars
            max_mortality = max([mortality.max() + sem.max() for _, mortality, sem in bar_data])
            axes[i].set_ylim(0, 100)

            # Add legend for populations
            #axes[i].legend(selected_populations[:max_populations_per_subplot], loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Adding common x-label to all axes in the last row
        for ax in axes[-ncol:]:  # Iterate over the last row
            ax.set_xlabel(xlabel, fontsize=16, fontweight='bold', fontname='Arial')
            ax.tick_params(axis='both', which='major', labelsize=14)  # Change font size for x-axis

            

        #plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
        plt.tight_layout()  # Adjust layout to make space for the legend
 
        return fig, axes  # Return the figure and axes
