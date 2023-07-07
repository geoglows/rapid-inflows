import glob
import os

import pandas as pd


def rapid_namelist(
        namelist_save_path: str,

        k_file: str,
        x_file: str,
        riv_bas_id_file: str,
        rapid_connect_file: str,
        vlat_file: str,
        qout_file: str,

        time_total: int,
        timestep_calc_routing: int,
        timestep_calc: int,
        timestep_inp_runoff: int,

        # Optional - Flags for RAPID Options
        run_type: int = 1,
        routing_type: int = 1,

        use_qinit_file: bool = False,
        qinit_file: str = '',  # qinit_VPU_DATE.csv

        write_qfinal_file: bool = True,
        qfinal_file: str = '',

        compute_volumes: bool = False,
        v_file: str = '',

        use_dam_model: bool = False,  # todo more options here
        use_influence_model: bool = False,
        use_forcing_file: bool = False,
        use_uncertainty_quantification: bool = False,

        opt_phi: int = 1,

        # Optional - Can be determined from rapid_connect
        reaches_in_rapid_connect: int = None,
        max_upstream_reaches: int = None,

        # Optional - Can be determined from riv_bas_id_file
        reaches_total: int = None,

        # Optional - Optimization Runs Only
        time_total_optimization: int = 0,
        timestep_observations: int = 0,
        timestep_forcing: int = 0,
) -> None:
    """
    Generate a namelist file for a RAPID routing run

    All units are strictly SI: meters, cubic meters, seconds, cubic meters per second, etc.

    Args:
        namelist_save_path (str): Path to save the namelist file
        k_file (str): Path to the k_file (input)
        x_file (str): Path to the x_file (input)
        rapid_connect_file (str): Path to the rapid_connect_file (input)
        qout_file (str): Path to save the Qout_file (routed discharge file)
        vlat_file (str): Path to the Vlat_file (inflow file)

    Returns:
        None
    """

    assert run_type in [1, 2], 'run_type must be 1 or 2'
    assert routing_type in [1, 2, 3, ], 'routing_type must be 1, 2, 3, or 4'
    assert opt_phi in [1, 2], 'opt_phi must be 1, or 2'

    if any([x is None for x in (reaches_in_rapid_connect, max_upstream_reaches)]):
        df = pd.read_csv(rapid_connect_file, header=None)
        reaches_in_rapid_connect = df.shape[0]
        rapid_connect_columns = ['rivid', 'next_down', 'count_upstream']  # plus 1 per possible upstream reach
        max_upstream_reaches = df.columns.shape[0] - len(rapid_connect_columns)

    if reaches_total is None:
        df = pd.read_csv(riv_bas_id_file, header=None)
        reaches_total = df.shape[0]

    namelist_options = {
        'BS_opt_Qfinal': f'.{str(write_qfinal_file).lower()}.',
        'BS_opt_Qinit': f'.{str(use_qinit_file).lower()}.',
        'BS_opt_dam': f'.{str(use_dam_model).lower()}.',
        'BS_opt_for': f'.{str(use_forcing_file).lower()}.',
        'BS_opt_influence': f'.{str(use_influence_model).lower()}.',
        'BS_opt_V': f'.{str(compute_volumes).lower()}.',
        'BS_opt_uq': f'.{str(use_uncertainty_quantification).lower()}.',

        'k_file': f"'{k_file}'",
        'x_file': f"'{x_file}'",
        'rapid_connect_file': f"'{rapid_connect_file}'",
        'riv_bas_id_file': f"'{riv_bas_id_file}'",
        'Qout_file': f"'{qout_file}'",
        'Vlat_file': f"'{vlat_file}'",
        'V_file': f"'{v_file}'",

        'IS_opt_run': run_type,
        'IS_opt_routing': routing_type,
        'IS_opt_phi': opt_phi,
        'IS_max_up': max_upstream_reaches,
        'IS_riv_bas': reaches_in_rapid_connect,
        'IS_riv_tot': reaches_total,

        'IS_dam_tot': 0,
        'IS_dam_use': 0,
        'IS_for_tot': 0,
        'IS_for_use': 0,

        'Qinit_file': f"'{qinit_file}'",
        'Qfinal_file': f"'{qfinal_file}'",

        'ZS_TauR': timestep_inp_runoff,
        'ZS_dtR': timestep_calc_routing,
        'ZS_TauM': time_total,
        'ZS_dtM': timestep_calc,
        'ZS_TauO': time_total_optimization,
        'ZS_dtO': timestep_observations,
        'ZS_dtF': timestep_forcing,
    }

    # generate the namelist file
    namelist_string = '\n'.join([
        '&NL_namelist',
        *[f'{key} = {value}' for key, value in namelist_options.items()],
        '/',
        ''
    ])

    with open(namelist_save_path, 'w') as f:
        f.write(namelist_string)


def rapid_namelist_from_directories(vpu_directory: str,
                                    inflows_directory: str,
                                    namelists_directory: str,
                                    outputs_directory: str, ) -> None:
    vpu_code = os.path.basename(vpu_directory)
    k_file = os.path.join(vpu_directory, f'k.csv')
    x_file = os.path.join(vpu_directory, f'x.csv')
    riv_bas_id_file = os.path.join(vpu_directory, f'riv_bas_id.csv')
    rapid_connect_file = os.path.join(vpu_directory, f'rapid_connect.csv')

    for x in (k_file, x_file, riv_bas_id_file, rapid_connect_file):
        assert os.path.exists(x), f'{x} does not exist'

    inflow_files = sorted(glob.glob(os.path.join(inflows_directory, '*.nc')))
    for idx, inflow_file in enumerate(inflow_files):
        start_date = os.path.basename(inflow_file).split('_')[2]
        end_date = os.path.basename(inflow_file).split('_')[3].replace('.nc', '')
        namelist_save_path = os.path.join(namelists_directory, f'rapid_namelist_{vpu_code}_{start_date}')
        vlat_file = inflow_file
        qout_file = os.path.join(outputs_directory, f'Qout_{vpu_code}_{start_date}_{end_date}.nc')  # todo

        time_total = ((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1) * 24 * 60 * 60
        timestep_calc_routing = 900
        timestep_calc = 24 * 60 * 60
        timestep_inp_runoff = 24 * 60 * 60

        write_qfinal_file: bool = True
        qfinal_file: str = os.path.join(outputs_directory, f'Qfinal_{vpu_code}_{end_date}.csv')

        use_qinit_file = idx > 0
        qinit_file = os.path.join(
            outputs_directory, f'Qfinal_{vpu_code}_{inflow_files[idx - 1].split("_")[-1]}.csv'
        ) if use_qinit_file else ''

        rapid_namelist(namelist_save_path=namelist_save_path,
                       k_file=k_file,
                       x_file=x_file,
                       riv_bas_id_file=riv_bas_id_file,
                       rapid_connect_file=rapid_connect_file,
                       vlat_file=vlat_file,
                       qout_file=qout_file,
                       time_total=time_total,
                       timestep_calc_routing=timestep_calc_routing,
                       timestep_calc=timestep_calc,
                       timestep_inp_runoff=timestep_inp_runoff,
                       write_qfinal_file=write_qfinal_file,
                       qfinal_file=qfinal_file,
                       use_qinit_file=use_qinit_file,
                       qinit_file=qinit_file, )

    return


if __name__ == '__main__':
    vpu_dirs = '/mnt/inputs'
    inflow_dirs = '/mnt/inflows'
    namelist_dirs = '/mnt/namelists'
    output_dirs = '/mnt/outputs'

    all_vpu_dirs = [x for x in glob.glob(os.path.join(vpu_dirs, '*')) if os.path.isdir(x)]
    for vpu_dir in all_vpu_dirs:
        inflow_dir = os.path.join(inflow_dirs, os.path.basename(vpu_dir))
        namelist_dir = os.path.join(namelist_dirs, os.path.basename(vpu_dir))
        output_dir = os.path.join(output_dirs, os.path.basename(vpu_dir))

        rapid_namelist_from_directories(vpu_directory=vpu_dir,
                                        inflows_directory=inflow_dir,
                                        namelists_directory=namelist_dir,
                                        outputs_directory=output_dir, )
