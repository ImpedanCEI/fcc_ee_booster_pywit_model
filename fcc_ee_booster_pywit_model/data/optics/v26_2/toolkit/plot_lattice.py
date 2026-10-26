"""
Plot lattice and optics functions
--------------------------------------------------------------------------------

Using the twiss file, this script plots the optics functions and layout.
Either beta-functions, dispersion, phases, or any combination thereof can be plotted.
Lattice plots can be displayed or saved as file.

*--Required--*
- **tfs_file** *(str)*: Path to the tfs file.

*--Optional--*
- **limits** *(floats)*: Limits on the x-axis.
- **optics** *(bool)*: Display beta functions.
- **dispersion** *(bool)*: Display dispersion.
- **phase** *(bool)*: Display phase.
- **show_plot** *(bool)*: Flag whether to display the plot.
- **plot_file** *(str)*: Filename to which the plot will be saved.
- **reference** *(str)*: To which point of an element the s-ccordinates refers to.

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
import tfs
from generic_parser import EntryPointParameters, entrypoint

# Constants --------------------------------------------------------------------
PLANES=('X', 'Y', 'Z')

PLOT_SUFFIX='.png'

ELEMENTSTYLE = {
    'SOL': {'color': 'yellow'},
    'SEXT': {'color': 'lime'},
    'SEXTUPOLE': {'color': 'lime'},
    'MULT': {'color': 'red'},
    'QUAD': {'color': 'red'},
    'QUADRUPOLE': {'color': 'red'},
    'BEND': {'color': 'midnightblue'},
    'SBEND': {'color': 'midnightblue'},
    'RBEND': {'color': 'midnightblue'},
}

REFER_SIGN = {
    "START":1,
    "CENTER":1,
    "END":-1,
}

# Script arguments -------------------------------------------------------------
def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="tfs_file",
        type=str,
        required=True,
        help="Path to the tfs file.",
    )
    params.add_parameter(
        name="limits",
        type=float,
        nargs=2,
        help="Limits on the x-axis.",
    )
    params.add_parameter(
        name="optics",
        action="store_true",
        help="Display beta functions.",
    )
    params.add_parameter(
        name="dispersion",
        action="store_true",
        help="Display dispersion.",
    )
    params.add_parameter(
        name="phase",
        action="store_true",
        help="Display phase.",
    )
    params.add_parameter(
        name="show_plot",
        action="store_true",
        help="Flag whether to display the plot.",
    )
    params.add_parameter(
        name="plot_file",
        type=str,
        default=None,
        help="Filename to which the plot will be saved.",
    )
    params.add_parameter(
        name="reference",
        type=str,
        choices=("START", "MIDDLE", "END"),
        default='END',
        help="To which point of an element the s-ccordinates refers to.",
    )    
    return params


# Entrypoint -------------------------------------------------------------------
@entrypoint(get_params(), strict=True)
def main(opt):
    opt = _check_opts(opt)
    optics_df = tfs.read(opt.tfs_file)
    _check_tfs_file(optics_df, opt.optics, opt.dispersion, opt.phase)
    return plot_lattice(optics_df,
                        {"optics":opt.optics,
                        "dispersion":opt.dispersion,
                        "phases":opt.phase},
                        opt.limits,
                        opt.plot_file,
                        opt.show_plot,
                        opt.reference)


def _check_opts(opt):

    if not opt.show_plot and opt.plot_file==None:
        raise ValueError("Lattice Plot is neither displayed nor saved anywhere.")

    if not (opt.optics or opt.dispersion or opt.phase):
        raise ValueError("No parameters to plot were selected.")

    if Path(opt.tfs_file).is_file():    
            opt["tfs_file"]=Path(opt.tfs_file)
    else:
        raise OSError("TFS file path appears to be invalid")
    
    if opt.limits != None and (opt.limits[0]>opt.limits[1]):
        opt["limits"]=[opt.limits[1], opt.limits[0]]

    opt=_convert_str_to_path(opt, "plot_file")
    
    return opt


def _convert_str_to_path(opt, file):
    if opt[file]!=None:
            opt[file]=Path(opt[file])
    return opt


def _check_tfs_file(tfs_file, optics, dispersion, phase):
    
    columns=tfs_file.columns
    if optics and not all([x in columns for x in ['BETX', 'BETY']]):
        raise ValueError('No optics functions found in the tfs-file.')
    if dispersion and not all([x in columns for x in ['DX', 'DY']]):
        raise ValueError('No dispersion functions found in the tfs-file.')
    if phase and not all([x in columns for x in ['MUX', 'MUY']]):
        raise ValueError('No phases found in the tfs-file.')


def plot_lattice(optics_df, plot_scenarios, limits, plot_file, show_plot, refer):

    number_of_subplots = sum([x for x in plot_scenarios.values()])

    fig, ax = plt.subplots(nrows=number_of_subplots+1,
                           ncols=1,
                           figsize=(9, 9),
                           constrained_layout=True,
                           sharex=True,
                           gridspec_kw={'height_ratios':[1]+[3]*number_of_subplots},
                           )
    
    ax[number_of_subplots].set_xlabel("s [m]", fontsize=16)
    if limits != None:
        ax[number_of_subplots].set_xlim(limits)
        optics_df=optics_df[optics_df['S'].between(limits[0], limits[1], inclusive=True)]

    add_magnets(ax[0], optics_df, refer)

    plot_counter=1
    if plot_scenarios['optics']:
        ax[plot_counter].plot(optics_df.S, optics_df.BETX, linewidth=2, color='red', label=r'$\beta_x$')
        ax[plot_counter].plot(optics_df.S, optics_df.BETY, linewidth=2, color='blue', label=r'$\beta_y$')
        ax[plot_counter].legend(fontsize=16)
        ax[plot_counter].set_ylim([0, None])
        ax[plot_counter].set_ylabel(r"$\beta~[m]$", fontsize=16)
        ax[plot_counter].tick_params(axis='both', which='major', labelsize=12)
        plot_counter+=1
    
    if plot_scenarios['dispersion']:
        ax[plot_counter].plot(optics_df.S, optics_df.DX, linewidth=2, color='lime', label=r'$D_x$')
        ax[plot_counter].legend(fontsize=16)
        ax[plot_counter].set_ylabel(r"$Dispersion$", fontsize=16)
        ax[plot_counter].tick_params(axis='both', which='major', labelsize=12)
        plot_counter+=1
    
    if plot_scenarios['phases']:
        ax[plot_counter].plot(optics_df.S, optics_df.MUX, linewidth=2, color='red', label=r'$\mu_x$')
        ax[plot_counter].plot(optics_df.S, optics_df.MUY, linewidth=2, color='blue', label=r'$\mu_y$')
        ax[plot_counter].legend(fontsize=16)
        ax[plot_counter].set_ylabel(r"$\mu~[2\pi]$", fontsize=16)
        ax[plot_counter].tick_params(axis='both', which='major', labelsize=12)
        plot_counter+=1

    if plot_file != None:
        plt.savefig(plot_file.with_suffix(PLOT_SUFFIX))
    if show_plot:
        plt.show()

    return fig, ax


def add_magnets(ax, data, refer):

    ax.axhline(color='black')
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim([-1.2, 1.2])
    
    for idx, elem in data.iterrows():
        try:
            ax.bar(elem['S']+REFER_SIGN[refer]*elem['L']/2.,
                   get_height(elem),
                   elem['L'],
                   bottom=get_bottom(elem),
                   color=ELEMENTSTYLE[str(elem['KEYWORD'])]['color'],
                   alpha=0.8)
        except KeyError:
            pass
    

def get_height(elem):
    if str(elem['KEYWORD']) == 'QUAD' or str(elem['KEYWORD']) == 'QUADRUPOLE':
        return np.sign(elem['K1L'])
    if str(elem['KEYWORD']) == 'MULT':
        return np.sign(elem['K1L'])
    elif str(elem['KEYWORD']) == 'BEND' or str(elem['KEYWORD']) == 'SBEND' or str(elem['KEYWORD']) == 'RBEND':
        return 1.
    elif str(elem['KEYWORD']) == 'SEXT' or str(elem['KEYWORD']) == 'SEXTUPOLE':
        return np.sign(elem['K2L'])*0.75
    else:
        return 0.


def get_bottom(elem):
    if str(elem['KEYWORD']) == 'QUAD' or str(elem['KEYWORD']) == 'QUADRUPOLE':
        return 0.
    elif str(elem['KEYWORD']) == 'MULT':
        return 0.
    elif str(elem['KEYWORD']) == 'BEND' or str(elem['KEYWORD']) == 'SBEND' or str(elem['KEYWORD']) == 'RBEND':
        return -0.5
    elif str(elem['KEYWORD']) == 'SEXT' or str(elem['KEYWORD']) == 'SEXTUPOLE':
        return 0.
    else:
        return 0.

# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()
