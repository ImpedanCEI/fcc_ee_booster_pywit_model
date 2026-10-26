"""
Plot aperture and beamsize
--------------------------------------------------------------------------------

Using an aperture file, this script plots the beamsize, aperture, and beam-stay-clear.


*--Required--*
- **tfs_file** *(str)*: Path to the tfs aperture file.

*--Optional--*
- **beamsize** *(bool)*: Display beamsize and and apertures.
- **sigma_for_momentum_acceptance** *(float)*: If given, the momentum acceptance for the specified sigma is calculated and plotted.
- **min_bsc** *(float)*: If specified, a line will be added in the bsc plot to illustrate the minimum allowed bsc.
- **limits** *(floats)*: Limits on the x-axis.
- **show_plot** *(bool)*: Flag whether to display the plot.
- **plot_file** *(str)*: Filename to which the plot will be saved.

"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt 
import tfs
from generic_parser import EntryPointParameters, entrypoint

# Constants --------------------------------------------------------------------

PLOT_SUFFIX='.png'
BSC_PLOT_MARGIN = 10
MOMENTUM_ACCEPTANCE_COLUMN = "Momentum_acceptance"
PLANES = ['X', 'Y']
PLANE_TO_APER = {'X': 'APER_1', 'Y': 'APER_2'}
UPPER_APERTURE_LIMIT = 0.04

# Script arguments -------------------------------------------------------------
def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="tfs_file",
        type=str,
        required=True,
        help="Path to the tfs aperture file.",
    )
    params.add_parameter(
        name="limits",
        type=float,
        nargs=2,
        help="Limits on the x-axis.",
    )
    params.add_parameter(
        name="beamsize",
        action="store_true",
        help="Display beamsize and and apertures.",
    )
    params.add_parameter(
        name="min_bsc",
        type=float,
        help="If specified, a line will be added in the bsc plot to illustrate the minimum allowed bsc.",
    )
    params.add_parameter(
        name="sigma_for_momentum_acceptance",
        type=float,
        help="If given, the momentum acceptance for the specified sigma is calculated and plotted.",
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
    return params


# Entrypoint -------------------------------------------------------------------
@entrypoint(get_params(), strict=True)
def main(opt):
    opt = _check_opts(opt)
    aperture_df = tfs.read(opt.tfs_file)
    return plot_lattice(aperture_df,
                        opt.beamsize,
                        opt.limits,
                        opt.min_bsc,
                        opt.sigma_for_momentum_acceptance,
                        opt.plot_file,
                        opt.show_plot)


def _check_opts(opt):

    if not opt.show_plot and opt.plot_file==None:
        raise ValueError("Aperture Plot is neither displayed nor saved anywhere.")

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


def calc_momentum_acceptance(df, n_sig):

    df[MOMENTUM_ACCEPTANCE_COLUMN] = (df['APER_1'] - n_sig*np.sqrt(df['BETX']*df.headers['EXN']/(df.headers['BETA']*df.headers['GAMMA'])) )/ abs(df['DX'])

    return df


def calc_n1min(df):
    n1min={}
    beta_gamma=df.headers['BETA']*df.headers['GAMMA']
    for plane in PLANES:
        n1min[plane] = np.min(
                              n1_formula(
                                          df[PLANE_TO_APER[plane]],
                                          df.headers['CO_RADIUS'] if plane == 'X' else 0.,
                                          df[f'{plane}TOL'],
                                          delta_disp(df, plane),
                                          df[f'BET{plane}'],
                                          df.headers['BETA_BEATING'],
                                          df.headers[f'E{plane}N'],
                                          beta_gamma
                                          )
                              )
                       

    return n1min

def delta_disp(df, plane):
    
    z_d = np.abs(df.headers["DP_BUCKET_SIZE"] * df[f'D{plane}'])
    
    z_spur = np.abs(df.headers["DP_BUCKET_SIZE"] 
            + df.headers["TWISS_DELTAP"]) * df.headers["DQF"] * df.headers[f"PARAS_D{plane}"] * np.sqrt(df[f"BET{plane}"]  / df.headers["BETAQFX"] )
    
    return df.headers["BETA_BEATING"] * (z_d + z_spur)


def n1_formula(aperture, closed_orbit_uncertainty, tolerance, spurious_dispersion, betafunction, beambeating, normalised_emittance, betagamma):
    return (aperture-closed_orbit_uncertainty-tolerance-spurious_dispersion)/np.sqrt(betafunction*normalised_emittance/betagamma)/beambeating


def plot_lattice(aperture_df, plot_beamsize, limits, min_bsc, mom_sig, plot_file, show_plot):

    plot_mom = True if mom_sig !=None else False
    fig, ax = plt.subplots(nrows=plot_beamsize*2+plot_mom+1,
                           ncols=1,
                           figsize=(12, 9),
                           constrained_layout=True,
                           sharex=True,
                           squeeze=False,
                           gridspec_kw={'height_ratios':[3]*(plot_beamsize*2+plot_mom)+[3]},
                           )
    
    if limits != None:
        ax[0,0].set_xlim(limits)

    n1min=calc_n1min(aperture_df[aperture_df.APER_1>0.0])
    beta_gamma=aperture_df.headers['BETA']*aperture_df.headers['GAMMA']

    plot_counter=0
    if plot_beamsize:
        
        ax[plot_counter, 0].step(aperture_df.S[aperture_df.APER_1>0.0], aperture_df.APER_1[aperture_df.APER_1>0.0], linewidth=2, color='grey', where='pre')

        ax[plot_counter, 0].plot(aperture_df.S, n1min['X']*np.sqrt(aperture_df.BETX*aperture_df.headers['EXN']/beta_gamma), linewidth=2, color='red', label=f'{n1min["X"]:.2f} $\sigma_x$')
        ax[plot_counter, 0].fill_between(aperture_df.S, y1=0, y2=n1min['X']*np.sqrt(aperture_df.BETX*aperture_df.headers['EXN']/beta_gamma), alpha=0.4, color='red')
    
        ax[plot_counter, 0].legend(fontsize=16, loc='upper right')
        ax[plot_counter, 0].set_ylabel(r"$x~[m]$", fontsize=16)
        ax[plot_counter, 0].set_ylim([0, UPPER_APERTURE_LIMIT])
        ax[plot_counter, 0].tick_params(axis='both', which='major', labelsize=12)
        plot_counter+=1
        
        ax[plot_counter, 0].step(aperture_df.S[aperture_df.APER_2>0.0],aperture_df.APER_2[aperture_df.APER_2>0.0], linewidth=2, color='grey', where='pre')

        ax[plot_counter, 0].plot(aperture_df.S,n1min['Y']*np.sqrt(aperture_df.BETY*aperture_df.headers['EYN']/beta_gamma), linewidth=2, color='blue', label=f'{n1min["Y"]:.2f} $\sigma_y$')
        ax[plot_counter, 0].fill_between(aperture_df.S, y1=0, y2=n1min['Y']*np.sqrt(aperture_df.BETY*aperture_df.headers['EYN']/beta_gamma), alpha=0.4, color='blue')
        
        ax[plot_counter, 0].legend(fontsize=16, loc='upper right')
        ax[plot_counter, 0].set_ylabel(r"$y~[m]$", fontsize=16)
        ax[plot_counter, 0].set_ylim([0, UPPER_APERTURE_LIMIT])
        ax[plot_counter, 0].tick_params(axis='both', which='major', labelsize=12)
        plot_counter+=1

    if plot_mom:
        
        aperture_df = calc_momentum_acceptance(aperture_df, mom_sig)

        ax[plot_counter, 0].plot(aperture_df.S[aperture_df.APER_1>0.0],
                                 aperture_df[MOMENTUM_ACCEPTANCE_COLUMN][aperture_df.APER_1>0.0],
                                 linewidth=2, color='darkblue', label=f'$(Aperture-{mom_sig:.1f}\sigma_x)/D_x$')
        
        ax[plot_counter, 0].set_ylim([0, 1.5*aperture_df[MOMENTUM_ACCEPTANCE_COLUMN][aperture_df.APER_1>0.0].min()])
        ax[plot_counter, 0].set_ylabel(r"$\delta_{max}$", fontsize=16)
        ax[plot_counter, 0].tick_params(axis='both', which='major', labelsize=12)
        ax[plot_counter, 0].legend(fontsize=16)
        
        print(f'Minimum momentum acceptance {aperture_df[MOMENTUM_ACCEPTANCE_COLUMN][aperture_df.APER_1>0.0].min(axis=0):.5f} '
              f'found at element {aperture_df.NAME.iloc[aperture_df[MOMENTUM_ACCEPTANCE_COLUMN][aperture_df.APER_1>0.0].idxmin(axis=0)]}, '
              f'which features a beam stay clear of {aperture_df.N1.iloc[aperture_df[MOMENTUM_ACCEPTANCE_COLUMN][aperture_df.APER_1>0.0].idxmin(axis=0)]:.2f}.')

        plot_counter+=1
    
    ax[plot_counter, 0].plot(aperture_df.S, aperture_df.N1, linewidth=2, color='red', label=f'min. beam stay clear\nfrom MAD-X: {aperture_df.headers["N1MIN"]:.2f}')
    ax[plot_counter, 0].set_xlabel("s [m]", fontsize=16)
    ax[plot_counter, 0].set_ylabel(r"Beam-stay-clear [$\sigma$]", fontsize=16)
    ax[plot_counter, 0].tick_params(axis='both', which='major', labelsize=12)

    ax[plot_counter, 0].legend(fontsize=16, loc='lower right')
    if min_bsc is not None:
        ax[plot_counter, 0].axhline(y=min_bsc, linewidth=2, color='lime')
    
    ax[plot_counter, 0].set_ylim([0, aperture_df.headers["N1MIN"]+BSC_PLOT_MARGIN])
    
    if plot_file != None:
        plt.savefig(plot_file.with_suffix(PLOT_SUFFIX))
    if show_plot:
        plt.show()

    return fig, ax


# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()
