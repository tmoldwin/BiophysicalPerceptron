from matplotlib import pyplot as plt

def axisConfigure(ax, numTicks = 4): 
    from matplotlib.ticker import MaxNLocator  as locator
    #plt.tick_params(axis='both',which='both',left = 'off' ,right = 'off', bottom='off',top='off')
    plt.gcf().subplots_adjust(bottom=0.15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.xaxis.set_major_locator( locator(numTicks) )
    ax.xaxis.set_minor_locator( locator(numTicks) )
    ax.yaxis.set_major_locator( locator(numTicks) )
    ax.yaxis.set_minor_locator( locator(numTicks, prune = 'lower') )
    ax.tick_params(axis='both', which='major', pad= 5)
    ax.margins(0)
    plt.tight_layout()