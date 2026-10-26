
from generic_parser import EntryPointParameters, entrypoint
import plot_aperture, plot_lattice
from pathlib import Path

REPOSITORY_TOP_LEVEL = Path(__file__).resolve().parent.parent

# Script arguments -------------------------------------------------------------
def get_params():
    params = EntryPointParameters()
    params.add_parameter(
        name="operation_mode",
        type=str,
        required=True,
        choices=['z', 'w', 'h', 't'],
        help="Which operation mode.",
    )
    return params

@entrypoint(get_params(), strict=True)
def main(opt):
    
    twiss_dir = REPOSITORY_TOP_LEVEL/'public'/opt.operation_mode
    plot_lattice.main(tfs_file=str(twiss_dir/f'twiss_{opt.operation_mode}_b1_nottapered.tfs'),
                      optics=True,
                      dispersion=True,
                      plot_file=str(twiss_dir/f'{opt.operation_mode}_lattice_optics.png'),
                      reference='END'
                      )
    plot_aperture.main(tfs_file=str(twiss_dir/f'aperture_{opt.operation_mode}_b1_nottapered.tfs'),
                       beamsize=True,
                       plot_file=str(twiss_dir/f'{opt.operation_mode}_aperture_all.png'),
                       sigma_for_momentum_acceptance=8.
                       )


    return 0

# Script Mode ------------------------------------------------------------------

if __name__ == "__main__":
    main()