"""FCC-ee HEB V26.2 single-bunch PyWIT model.

Install beside fcc_ee_booster_pywit_model.py using install_hebmodel.py.

No CST numerical data and no Patrick factor enter the resistive-wall response.
The parameter q=-2.1043 is a CST-equivalent *reference convention* found
empirically (its underlying physical derivation is not established).
Four-layer RW / coupled-bunch modeling is explicitly outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import os
import re
import subprocess
import warnings
import zipfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.constants import c, mu_0, epsilon_0, physical_constants
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

from pywit.component import Component
from pywit.element import Element
from pywit.model import Model

Z0 = mu_0 * c
PLANES = ('long', 'x', 'y')
SHORTHAND = {'long':'z0000', 'x':'x1000', 'y':'y0100'}
FAMILIES = ('RW', 'Wiggler', 'Kicker', 'RF', 'BPM', 'Bellows')
SOURCE_PATHS = {
    'RF': {'long':'rf_cavities/Z_single_rf_cavity_long_filter_3mm.txt',
           'x':'rf_cavities/Z_single_rf_cavity_dipx_filter_3mm.txt',
           'y':'rf_cavities/Z_single_rf_cavity_dipy_filter_3mm.txt'},
    'BPM':{'long':'bpm/Z_bpm_long_3mm.txt',
           'x':'bpm/Z_bpm_dipx_3mm.txt',
           'y':'bpm/Z_bpm_dipy_3mm.txt'},
    'Bellows':{'long':'bellows/Z_bellows_long.txt',
               'x':'bellows/Z_bellows_dipx.txt',
               'y':'bellows/Z_bellows_dipy.txt'},
}
KICKER_FILES = {
    'long':'Z_long_optimizedKicker.txt',
    'x':'Z_dipx_optimizedKicker.txt',
    'y':'Z_dipy_optimizedKicker.txt',
}


def gamma_eta(f_hz, sigma):
    f = np.atleast_1d(np.asarray(f_hz, float))
    w = 2*np.pi*f
    gamma = np.sqrt(1j*w*mu_0*(sigma+1j*w*epsilon_0))
    eta = np.zeros_like(gamma, dtype=complex)
    np.divide(1j*w*mu_0, gamma, out=eta, where=np.abs(gamma)>0)
    return gamma, eta


def tl_layer(z_load, f_hz, sigma, thickness_m):
    gamma, eta = gamma_eta(f_hz, sigma)
    u = np.tanh(gamma*thickness_m)
    den = eta+z_load*u
    z = np.zeros_like(eta, complex)
    np.divide(eta*(z_load+eta*u), den, out=z, where=np.abs(den)>0)
    return z


def gold_copper_surface(f_hz, *, sigma_cu, rq_m, q=-2.1043,
                        points_per_rq=300, deep_span_rq=8):
    """Gaussian-gradient TL, E/H at x=q*Rq with copper bulk load.

    sigma(x) = sigma_cu * Phi(x/Rq), x increasing into the copper.
    """
    f = np.atleast_1d(np.asarray(f_hz, float))
    if np.any(f <= 0):
        raise ValueError('Gold surface response must be evaluated at f>0')
    if rq_m<=0 or points_per_rq<40 or deep_span_rq<q+6:
        raise ValueError('Invalid roughness grid/span')
    n = max(1, int(np.ceil((deep_span_rq-q)*points_per_rq)))
    edges = np.linspace(q, deep_span_rq, n+1)*rq_m
    sigma_steps = sigma_cu*ndtr(0.5*(edges[1:]+edges[:-1])/rq_m)
    z = gamma_eta(f, sigma_cu)[1]
    for sig, thickness in zip(sigma_steps[::-1], np.diff(edges)[::-1]):
        z = tl_layer(z, f, float(sig), float(thickness))
    return z


def neg_cu_surface(f_hz, *, sigma_cu=5.8e7, sigma_neg=1e6, t_neg_m=150e-9,
                   rq_m=0.3e-6, q=-2.1043, points_per_rq=300,
                   deep_span_rq=8, rough=True):
    f = np.atleast_1d(np.asarray(f_hz, float))
    positive = f>0
    out = np.zeros(len(f), complex)
    if not np.any(positive): return out
    if rough:
        zcu = gold_copper_surface(f[positive], sigma_cu=sigma_cu, rq_m=rq_m,
                                 q=q, points_per_rq=points_per_rq,
                                 deep_span_rq=deep_span_rq)
    else:
        zcu = gamma_eta(f[positive],sigma_cu)[1]
    out[positive] = tl_layer(zcu, f[positive], sigma_neg, t_neg_m)
    return out


def circular_rw_per_m(f_hz, zeta, b_m):
    """Ultra-relativistic round-wall impedance, per 1 m longitudinal length.

    Z_parallel returned Ω/m length; Z_dipolar returned Ω/m transverse per m length.
    Matches the conventional Z_dip ∝ c ζ/(ωπb³) form used in HEB benchmark.
    """
    f=np.atleast_1d(np.asarray(f_hz,float))
    zeta=np.atleast_1d(np.asarray(zeta,complex))
    w=2*np.pi*f
    k=w/c
    kb=k*b_m
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        zl=(zeta/(2*np.pi*b_m))/(1+1j*(zeta/Z0)*kb/2)
        zd=(c*zeta/(w*np.pi*b_m**3))/(1+1j*(zeta/Z0)*(kb/2-1/kb)+(zeta/Z0)**2)
    return np.nan_to_num(zl),np.nan_to_num(zd)


def read_complex_impedance(path_or_file, *, name='component'):
    """Require F[GHz], ReZ, ImZ. Reject magnitude-only sources."""
    try:
        raw=np.loadtxt(path_or_file, comments='#', usecols=(0,1,2), ndmin=2)
    except (ValueError, IndexError) as exc:
        raise ValueError(f'{name} requires THREE columns: frequency [GHz], ReZ, ImZ; '
                         'magnitude-only files cannot replace complex impedance') from exc
    if raw.ndim!=2 or raw.shape[1]!=3 or len(raw)<3:
        raise ValueError(f'{name} invalid three-column table')
    if not np.all(np.isfinite(raw)):
        raise ValueError(f'{name} contains non-finite values')
    idx=np.argsort(raw[:,0]); raw=raw[idx]
    raw=raw[np.r_[True,np.diff(raw[:,0])>0]]
    if raw[0,0]<0 or raw[-1,0]<=0:
        raise ValueError(f'{name} requires nonnegative frequencies')
    return raw[:,0]*1e9, raw[:,1]+1j*raw[:,2]


def load_kicker(data_dir, *, archive=None):
    """Read complete 20 m optimized kicker, from files or nested ZIP.

    Does NOT silently choose a different source, apply extra longitudinal length,
    or include quadrupolar impedance.
    """
    folder=Path(data_dir)/'kickers'
    found={}
    if folder.is_dir():
        files=[p for p in folder.rglob('*') if p.is_file()]
        for plane,fn in KICKER_FILES.items():
            matches=[p for p in files if p.name.lower()==fn.lower()]
            if len(matches)>1:
                raise RuntimeError(f'Multiple candidates for kicker {fn}: {matches}')
            if len(matches)==1:found[plane]=matches[0]
    if len(found)==3:
        return {p:(read_complex_impedance(fp,name=f'kicker {p}'),str(fp))
                for p,fp in found.items()}
    candidates=[]
    if archive is not None:candidates.append(Path(archive).expanduser())
    candidates += [folder/'kicker_z_w.zip',Path(data_dir)/'kicker_z_w.zip',
                   Path.cwd()/'kicker_z_w.zip']
    if folder.is_dir():candidates += sorted(folder.glob('*.zip'))
    seen=set()
    for zp in candidates:
        if not zp.is_file() or zp.resolve() in seen:continue
        seen.add(zp.resolve())
        if not zipfile.is_zipfile(zp):continue
        with zipfile.ZipFile(zp) as arc:
            mapping={}
            for pl,fn in KICKER_FILES.items():
                choices=[m for m in arc.namelist() if Path(m).name.lower()==fn.lower()]
                if len(choices)==1:mapping[pl]=choices[0]
            if len(mapping)!=3:continue
            out={}
            for pl,m in mapping.items():
                with arc.open(m) as fh:
                    out[pl]=(read_complex_impedance(fh,name=f'kicker {pl}'),f'{zp}::{m}')
            return out
    raise FileNotFoundError(
        'Kicker files not found. Expected three Z_*optimizedKicker.txt files, '
        f'under {folder} (nested allowed), or data/kickers/kicker_z_w.zip. '
        'Set kicker_archive=... if the supplied ZIP is elsewhere.')


def interpolated_complex(f_src,z_src, f_dst):
    """Real/imag interpolation, ZERO outside physical file bandwidth."""
    f_src=np.asarray(f_src,float);z_src=np.asarray(z_src,complex)
    f_dst=np.asarray(f_dst,float)
    return (np.interp(f_dst,f_src,z_src.real,left=0,right=0)
            +1j*np.interp(f_dst,f_src,z_src.imag,left=0,right=0))


def flat_iw2d_copper(*, cache_path, half_gap_m=0.01, rho_cu=1.66e-8,
                      gamma=20e9/(510998.95), fmin_exp=-1, fmax_exp=11,
                      points_per_decade=60, executable=None):
    """Flat-chamber IW2D wiggler per 1 m, using original file/cache conventions."""
    cache_path=Path(cache_path).expanduser()
    if cache_path.is_file():
        with np.load(cache_path) as obj:
            d={key:obj[key] for key in obj.files}
        if not {'fz','Zz','fx','Zx','fy','Zy'}.issubset(d):
            raise ValueError(f'Wiggler cache missing required fields: {cache_path}')
        return {'long':(d['fz'],d['Zz']), 'x':(d['fx'],d['Zx']),
                'y':(d['fy'],d['Zy'])},str(cache_path)
    if executable is None:
        env=os.environ.get('HEB_IW2D_FLATCHAMBER','')
        possible=([Path(env).expanduser()] if env else [])+[
            Path.home()/'pywit/pywit/IW2D/IW2D/cpp/flatchamber.x',
            Path.home()/'pywit/IW2D/cpp/flatchamber.x',
            Path.home()/'pywit/IW2D/bin/flatchamber.x',
            Path.home()/'pywit/IW2D/IW2D/cpp/flatchamber.x']
        executable=next((p for p in possible if p.is_file()),None)
    if executable is None:
        raise FileNotFoundError('Missing flat-chamber IW2D for wiggler (not circular RW). '
                                f'No validated cache at {cache_path}; '
                                'set HEB_IW2D_FLATCHAMBER to flatchamber.x or supply wiggler_cache=...')
    executable=Path(executable).expanduser()
    job_dir=cache_path.parent/('run_'+cache_path.stem)
    job_dir.mkdir(parents=True,exist_ok=True)
    inp='\n'.join([
        'Machine:\tFCCeeHEB',f'Relativistic Gamma:\t{gamma:.15g}',
        'Impedance Length in m:\t1.0',
        'Maximum order of non-linear terms:\t0',
        'Number of upper layers in the chamber wall:\t1',
        f'Layer 1 inner half gap in mm:\t{half_gap_m*1e3:.15g}',
        f'Layer 1 DC resistivity (Ohm.m):\t{rho_cu:.15g}',
        'Layer 1 relaxation time for resistivity (ps):\t0.0',
        'Layer 1 real part of dielectric constant:\t1.0',
        'Layer 1 magnetic susceptibility:\t0.0',
        'Layer 1 relaxation frequency of permeability (MHz):\tInfinity',
        'Layer 1 thickness in mm:\tInfinity',
        'Top bottom symmetry (yes or no):\tyes',
        f'start frequency exponent (10^) in Hz:\t{fmin_exp}',
        f'stop frequency exponent (10^) in Hz:\t{fmax_exp}',
        'linear (1) or logarithmic (0) or both (2) frequency scan:\t0',
        f'Number of points per decade (for log):\t{int(points_per_decade)}',
        'when both, fmin of the refinement (in THz):\t1.0',
        'when both, fmax of the refinement (in THz):\t10.0',
        'when both, number of points in the refinement:\t100',
        'added frequencies (Hz):\t',
        f'Comments for the output files names:\t_{cache_path.stem}',
    ])+'\n'
    for old in job_dir.glob('Z*.dat'):old.unlink()
    def run(text):
        return subprocess.run([str(executable)],input=text,text=True,cwd=job_dir,
                              capture_output=True)
    result=run(inp)
    if result.returncode!=0:
        inp=inp.replace('added frequencies (Hz):','added frequencies [Hz]:')
        result=run(inp)
    if result.returncode!=0:
        raise RuntimeError('IW2D flatchamber.x failed: '+result.stderr[-2000:])
    def read(pattern):
        found=sorted(job_dir.glob(pattern),key=lambda x:x.stat().st_mtime)
        if not found:raise FileNotFoundError(f'Expected IW2D result {pattern} under {job_dir}')
        rows=[]
        for ln in found[-1].read_text(errors='replace').splitlines():
            fields=ln.split()
            if len(fields)<3:continue
            try:row=tuple(float(x) for x in fields[:3])
            except ValueError:continue
            if np.all(np.isfinite(row)):rows.append(row)
        if len(rows)<3:raise ValueError(f'Malformed IW2D output: {found[-1]}')
        d=np.asarray(rows)
        return d[:,0],d[:,1]+1j*d[:,2]
    d={}
    for pl,pattern in [('long','Zlong*.dat'),('x','Zxdip*.dat'),('y','Zydip*.dat')]:
        d[pl]=read(pattern)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path, fz=d['long'][0],Zz=d['long'][1],
                        fx=d['x'][0],Zx=d['x'][1],fy=d['y'][0],Zy=d['y'][1])
    return d,str(cache_path)


def _resolve_data_dir(data_dir=None):
    """Use the *installed local HEB package* data root; never a GitHub cache."""
    if data_dir is not None:
        candidate=Path(data_dir).expanduser().resolve()
        if (candidate/'data'/'rf_cavities').is_dir():candidate=candidate/'data'
        if not (candidate/'rf_cavities').is_dir():
            raise FileNotFoundError(f'Explicit data_dir is not a package data directory: {candidate}')
        return candidate
    try:
        from .package_paths import base_dir
        root=Path(base_dir).expanduser().resolve()
    except ImportError:
        root=Path(__file__).resolve().parent
    candidates=[root/'data',root/'fcc_ee_booster_pywit_model'/'data',
                Path(__file__).resolve().parent/'data',root]
    for d in candidates:
        if (d/'rf_cavities').is_dir() and (d/'bpm').is_dir():return d
    raise FileNotFoundError('Cannot locate the installed HEB package data/ tree. '
                            'Use data_dir=... pointing to your LOCAL fcc_ee_booster_pywit_model/data.')


def _interp_log_curves(fsrc,zsrc,fdst):
    fsrc=np.asarray(fsrc,float);zsrc=np.asarray(zsrc,complex)
    fdst=np.atleast_1d(np.asarray(fdst,float))
    out=np.zeros(len(fdst),complex)
    m=(fdst>=fsrc[0])&(fdst<=fsrc[-1])
    if np.any(m):
        x=np.log(fsrc)
        xin=np.log(fdst[m])
        out[m]=(PchipInterpolator(x,zsrc.real)(xin)
                +1j*PchipInterpolator(x,zsrc.imag)(xin))
    return out


@dataclass(frozen=True)
class HEBSourceRecord:
    family: str
    plane: str
    path: str
    normalization: str
    multiplier: float
    input_fmax_ghz: float


class HEBModel(Model):
    """V26.2 HEB impedance model built from native PyWIT Components / Elements.

    Replaces the IW2D-only HEBModel class in the installed project.
    Distributed circular RW: Gold Gaussian copper + 150 nm NEG TL, to 10 THz.
    Damping wiggler: separate original flat-chamber IW2D (or validated cache).
    Other elements: exact local RF/BPM/bellows 3-column inputs and supplied kicker.

    Call `HEBModel(...).impedance_budget(...)`, `.bunch_wake_04mm(...)`.
    The PyWIT interface is `model.total.get_component('z0000')`, etc.
    All six Element objects are *ring-integrated* and beta-weighted beforehand.
    They have length=1, beta=reference beta to prevent double normalization.
    Their physical length and placement are instead stored in model.inventory,
    model.locations, and model.source_provenance.
    """
    def __init__(self, energy=45.6e9, optics_filename=None, yaml_file=None,
                 n_rf=112, rf_name_hint='ac800', *, data_dir=None,
                 sigma_cu=5.8e7, sigma_neg=1e6, neg_t_m=150e-9,
                 roughness_rq_m=0.3e-6, gold_reference_q=-2.1043,
                 beam_pipe_radius_m=0.030, gold_points_per_rq=300,
                 rw_fmax_hz=10e12, rw_nfreq=2801,
                 kicker_length_m=4.18, kicker_reference_length_m=20.0,
                 kicker_archive=None, kicker_s_positions_m=None,
                 kicker_beta_x_m=None, kicker_beta_y_m=None,
                 wigglers=10, wiggler_each_length_m=6.03,
                 wiggler_gap_m=0.020, wiggler_beta_m=4.0,
                 wiggler_s_positions_m=None, wiggler_cache=None,
                 wiggler_iw2d_executable=None, wiggler_source=None,
                 n_bellows=10000, bellows_length_m=0.15,
                 beta_reference='layout', require_v262=True):
        del yaml_file  # kept in signature for compatibility; circular RW no longer invokes IW2D
        self.machine='FCCee_HEB'
        self.energy=float(energy)
        self.relativistic_gamma=self.energy/(physical_constants['electron mass energy equivalent in MeV'][0]*1e6)
        self.relativistic_beta=np.sqrt(1-1/self.relativistic_gamma**2)
        self.data_dir=_resolve_data_dir(data_dir)
        if optics_filename is None:
            optics_filename=self.data_dir/'optics'/'v26_2'/'heb_ring_z.tfs'
        self.optics_filename=str(Path(optics_filename).expanduser().resolve())
        if not Path(self.optics_filename).is_file():
            raise FileNotFoundError(f'HEB V26.2 TFS not found: {self.optics_filename}. '
                                    'Specify optics_filename=... using your existing LOCAL V26.2 file.')
        if require_v262 and 'v26_2' not in self.optics_filename.lower():
            warnings.warn('Optics path does not contain v26_2: verify the exact lattice version',stacklevel=2)
        from .fcc_ee_booster_pywit_model import read_tfs_as_twiss_like
        self.twiss=read_tfs_as_twiss_like(self.optics_filename)
        self.twiss_df=self.twiss._df.copy()
        self.circ=float(self.twiss.summary.length)
        if not np.isfinite(self.circ):self.circ=float(np.nanmax(self.twiss.s))
        if not (self.circ>1000):raise ValueError('Invalid HEB circumference in TFS')
        self.q_x=float(self.twiss.summary.q1)
        self.q_y=float(self.twiss.summary.q2)
        rad=self.circ/(2*np.pi)
        self.beta_x_smooth=rad/self.q_x if np.isfinite(self.q_x) else np.nan
        self.beta_y_smooth=rad/self.q_y if np.isfinite(self.q_y) else np.nan
        self.beta_reference_mode=str(beta_reference)
        if self.beta_reference_mode not in ('layout','twiss_integral'):
            raise ValueError("beta_reference must be 'layout' or 'twiss_integral'")

        df=self.twiss_df
        name_col=next((a for a in df if str(a).upper()=='NAME'),None)
        s_col=next((a for a in df if str(a).upper()=='S'),None)
        bx_col=next((a for a in df if str(a).upper() in ('BETX','BETAX')),None)
        by_col=next((a for a in df if str(a).upper() in ('BETY','BETAY')),None)
        l_col=next((a for a in df if str(a).upper()=='L'),None)
        if None in (name_col,s_col,bx_col,by_col):
            raise ValueError('TFS must supply NAME, S, BETX, BETY')
        v=pd.DataFrame({'name':df[name_col].astype(str).str.strip().str.strip('"'),
                        's_m':pd.to_numeric(df[s_col],errors='coerce'),
                        'beta_x_m':pd.to_numeric(df[bx_col],errors='coerce'),
                        'beta_y_m':pd.to_numeric(df[by_col],errors='coerce')})
        v['length_m']=pd.to_numeric(df[l_col],errors='coerce') if l_col else 0.
        self.optics_table=v.dropna(subset=['s_m','beta_x_m','beta_y_m']).sort_values('s_m')
        line=self.optics_table.drop_duplicates('s_m')
        sline=line.s_m.to_numpy(float)
        if len(line)<3 or not np.all(np.diff(sline)>0):raise ValueError('TFS s coordinates not suitable for beta weighting')
        self.beta_x_twiss_avg=float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(line.beta_x_m,sline)/(sline[-1]-sline[0]))
        self.beta_y_twiss_avg=float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(line.beta_y_m,sline)/(sline[-1]-sline[0]))

        self.element_names=[]
        self.beta_xs=[];self.beta_ys=[];self.lengths=[]
        # Preserve machine-layout interpretation of the original HEBModel
        if beta_reference=='layout':
            from .utils import compute_betas_and_lengths
            from .data.machine_layouts.fcc_ee_heb_layout_b1 import layout_dict
            layout=compute_betas_and_lengths(twiss_table=self.twiss,layout_dict=layout_dict)
            keys=[key for key in layout if 'pipe' in key.lower()]
            if not keys:
                raise RuntimeError('Original HEB machine layout produced no pipe elements. '
                                   'Inspect compute_betas_and_lengths() and layout_dict.')
            self.element_names=keys
            self.beta_xs=[float(layout[k]['beta_x']) for k in keys]
            self.beta_ys=[float(layout[k]['beta_y']) for k in keys]
            self.lengths=[float(layout[k]['length']) for k in keys]
            seg_total=sum(self.lengths)
            if seg_total<=0:raise ValueError('Nonpositive layout pipe length')
            self.beta_x_avg=sum(x*l for x,l in zip(self.beta_xs,self.lengths))/seg_total
            self.beta_y_avg=sum(y*l for y,l in zip(self.beta_ys,self.lengths))/seg_total
        else:
            self.beta_x_avg=self.beta_x_twiss_avg
            self.beta_y_avg=self.beta_y_twiss_avg
        if not (self.beta_x_avg>0 and self.beta_y_avg>0):
            raise ValueError('Non-positive average beta')
        self.beta_x_ref=self.beta_x_avg
        self.beta_y_ref=self.beta_y_avg

        self.rf_locations=self.optics_table[self.optics_table.name.str.contains(rf_name_hint,case=False,regex=False)].copy()
        self.bpm_beta_table=self.optics_table[self.optics_table.name.str.match(r'(?i)^bpm\.',na=False)].copy()
        self.n_rf=len(self.rf_locations)
        self.n_bpm=len(self.bpm_beta_table)
        if self.n_rf!=n_rf:
            raise ValueError(f'RF mismatch: found {self.n_rf} TFS rows matching {rf_name_hint!r}, '
                             f'but n_rf={n_rf}; select the correct RF name hint.')
        if not self.n_bpm:raise ValueError('No named BPMs in V26.2 TFS')
        self.rf_beta_record={key:self.rf_locations.iloc[0][key] for key in self.rf_locations.columns}
        self.rf_beta_factor_x=float(self.rf_locations.beta_x_m.sum()/self.beta_x_avg)
        self.rf_beta_factor_y=float(self.rf_locations.beta_y_m.sum()/self.beta_y_avg)
        self.bpm_beta_factor_x=float(self.bpm_beta_table.beta_x_m.sum()/self.beta_x_avg)
        self.bpm_beta_factor_y=float(self.bpm_beta_table.beta_y_m.sum()/self.beta_y_avg)
        self.bellows_beta_factor_x=self.bellows_beta_factor_y=1.0
        self.n_bellows=int(n_bellows)
        self.bellows_length_m=float(bellows_length_m)
        self.wiggler_count=int(wigglers)
        self.wiggler_each_length_m=float(wiggler_each_length_m)
        self.wiggler_total_length_m=self.wiggler_count*self.wiggler_each_length_m
        self.wiggler_gap_m=float(wiggler_gap_m)
        self.wiggler_beta_m=float(wiggler_beta_m)
        self.kicker_length_m=float(kicker_length_m)
        self.kicker_reference_length_m=float(kicker_reference_length_m)
        self.kicker_factor=self.kicker_length_m/self.kicker_reference_length_m
        if not (0<self.kicker_factor<=1):raise ValueError('Unphysical HEB/reference kicker lengths')
        self.rw_length_m=(self.circ-self.n_bellows*self.bellows_length_m-
                          self.wiggler_total_length_m-self.kicker_length_m)
        if self.rw_length_m<=0:raise ValueError('Negative RW length after subtracting inserts')

        self.wiggler_locations=self._wiggler_locations(wiggler_s_positions_m)
        self.wiggler_beta_length_factor_x=self.wiggler_total_length_m*self.wiggler_beta_m/self.beta_x_avg
        self.wiggler_beta_length_factor_y=self.wiggler_total_length_m*self.wiggler_beta_m/self.beta_y_avg
        self.kicker_locations=self._kicker_locations(kicker_s_positions_m,kicker_beta_x_m,kicker_beta_y_m)
        if self.kicker_locations.empty:
            self.kicker_beta_factor_x=1.0
            self.kicker_beta_factor_y=1.0
            self.kicker_beta_source='ring-average (provisional; kicker s not supplied)'
        else:
            self.kicker_beta_factor_x=float(self.kicker_locations.beta_x_m.mean()/self.beta_x_avg)
            self.kicker_beta_factor_y=float(self.kicker_locations.beta_y_m.mean()/self.beta_y_avg)
            self.kicker_beta_source='local beta from TFS or supplied values'

        self.sigma_cu=float(sigma_cu)
        self.sigma_neg=float(sigma_neg)
        self.neg_t_m=float(neg_t_m)
        self.rq_m=float(roughness_rq_m)
        self.gold_q=float(gold_reference_q)
        self.b_m=float(beam_pipe_radius_m)
        self.rw_fmax_hz=float(rw_fmax_hz)
        self._rw_log_grid=np.geomspace(1e3,self.rw_fmax_hz,int(rw_nfreq))
        self._rw_surface=neg_cu_surface(self._rw_log_grid,sigma_cu=self.sigma_cu,
                            sigma_neg=self.sigma_neg,t_neg_m=self.neg_t_m,
                            rq_m=self.rq_m,q=self.gold_q,
                            points_per_rq=int(gold_points_per_rq))
        if not np.all(np.isfinite(self._rw_surface)) or np.any(self._rw_surface.real<0):
            raise ValueError('Gold/TL RW surface impedance failed positivity/finite checks')
        zlong,ztd=circular_rw_per_m(self._rw_log_grid,self._rw_surface,self.b_m)
        self._rw_samples={'long':zlong*self.rw_length_m,
                          'x':ztd*self.rw_length_m,
                          'y':ztd*self.rw_length_m}

        self._family_tables={}
        self.source_records=[]
        for family, paths in SOURCE_PATHS.items():
            self._family_tables[family]={}
            for pl,relative in paths.items():
                path=self.data_dir/relative
                if not path.is_file():
                    extra=(' The GitHub fork contains only Z_abs_single_rf_cavity_long_3mm.txt; '
                           'a magnitude cannot substitute for the missing complex RF impedance.'
                           if family=='RF' and pl=='long' else '')
                    raise FileNotFoundError(f'Missing {family}/{pl} input: {path}.{extra}')
                f,z=read_complex_impedance(path,name=f'{family}/{pl}')
                self._family_tables[family][pl]=(f,z)
                multi=self._family_multiplier(family,pl)
                norm=('single element, location-beta weighted' if family in ('RF','BPM')
                      else 'already integrated complete ring; NO extra count')
                self.source_records.append(HEBSourceRecord(family,pl,str(path),norm,multi,f[-1]/1e9))
        ki=load_kicker(self.data_dir,archive=kicker_archive)
        self._family_tables['Kicker']={}
        for pl in PLANES:
            (f,z),path=ki[pl]
            self._family_tables['Kicker'][pl]=(f,z)
            self.source_records.append(HEBSourceRecord('Kicker',pl,path,
                     '20 m source, scaled once to 4.18 m HEB system; local beta provisional',
                     self._family_multiplier('Kicker',pl),f[-1]/1e9))

        if wiggler_source is not None:
            wd={pl:wiggler_source[pl] for pl in PLANES}
            wsource='explicit supplied wiggler arrays (1 m reference)'
        else:
            if wiggler_cache is None:
                wiggler_cache=Path.home()/'pywit'/'iw2d_cache'/'fcc_ee_heb_v26_2_wiggler'/\
                    'wiggler_config_A_budget_e-1_to_e11_ppd60.npz'
            wd,wsource=flat_iw2d_copper(cache_path=wiggler_cache,
                         half_gap_m=self.wiggler_gap_m/2,
                         executable=wiggler_iw2d_executable)
        self._family_tables['Wiggler']=wd
        for pl in PLANES:
            f,z=wd[pl]
            self.source_records.append(HEBSourceRecord('Wiggler',pl,str(wsource),
                 'flat IW2D per metre, length/beta weighted once',
                 self._family_multiplier('Wiggler',pl),float(f[-1]/1e9)))
        for pl in PLANES:
            self.source_records.append(HEBSourceRecord('RW',pl,'Gold Gaussian Cu + NEG TL (computed)',
                 'CST-equivalent reference q=-2.1043, one circular pipe length',
                 self.rw_length_m,float(self._rw_log_grid[-1]/1e9)))
        self.source_provenance=pd.DataFrame([vars(x) for x in self.source_records])

        self.inventory=pd.DataFrame([
            {'component':'RW','count':1,'physical_length_m':self.rw_length_m,
             'location_status':'distributed in ring (excludes bellows, wiggler, kicker)'},
            {'component':'Wiggler','count':self.wiggler_count,'physical_length_m':self.wiggler_total_length_m,
             'location_status':'provisional symmetric S4; target beta 4 m'},
            {'component':'Kicker','count':10,'physical_length_m':self.kicker_length_m,
             'location_status':self.kicker_beta_source},
            {'component':'RF','count':self.n_rf,'physical_length_m':np.nan,
             'location_status':'named V26.2 ac800 cavities'},
            {'component':'BPM','count':self.n_bpm,'physical_length_m':np.nan,
             'location_status':'named V26.2 BPMs'},
            {'component':'Bellows','count':self.n_bellows,
             'physical_length_m':self.n_bellows*self.bellows_length_m,
             'location_status':'positions not available; source already ring integrated'},
        ])
        # The six *actual* PyWIT Element objects have ring-normalized impedances.
        # All use exactly the reference betas: PyWIT Model.total must NOT apply
        # another sqrt(beta) ratio to the already beta-weighted spectra.
        elems=[]
        for family in FAMILIES:
            comps=[]
            for pl,plane,exponents in [('long','z',(0,0)),('x','x',(1,0)),('y','y',(0,1))]:
                def zfun(freq,group=family,channel=pl):
                    return self.component_impedance(group,channel,freq)
                comps.append(Component(impedance=zfun,plane=plane,
                              source_exponents=exponents,test_exponents=(0,0),
                              name=f'{family}_{pl}'))
            elems.append(Element(length=1.0,beta_x=self.beta_x_avg,
                         beta_y=self.beta_y_avg,components=comps,
                         name=family,description='Ring integrated + beta weighted in HEBModel'))
        super().__init__(elements=elems,lumped_betas=(self.beta_x_avg,self.beta_y_avg))

    def _family_multiplier(self,family,plane):
        if family=='RF':
            return self.n_rf if plane=='long' else getattr(self,f'rf_beta_factor_{plane}')
        if family=='BPM':
            return self.n_bpm if plane=='long' else getattr(self,f'bpm_beta_factor_{plane}')
        if family=='Bellows':return 1.
        if family=='Kicker':
            return self.kicker_factor*(1 if plane=='long' else getattr(self,f'kicker_beta_factor_{plane}'))
        if family=='Wiggler':
            return self.wiggler_total_length_m if plane=='long' else getattr(self,f'wiggler_beta_length_factor_{plane}')
        if family=='RW':return self.rw_length_m
        raise KeyError(family)

    def _wiggler_locations(self, supplied_positions):
        if supplied_positions is None:
            df=self.optics_table
            ip=df[df.name.str.match(r'(?i)^IP[_\.]?4$',na=False)]
            if ip.empty:
                ip=df[df.name.str.contains(r'(?i)^IP[_\.]?4',na=False,regex=True)]
            if ip.empty:
                raise ValueError('Cannot locate IP4 for provisional S4 wiggler insertion')
            ip4=float(ip.iloc[0].s_m)
            slots=df[df.name.str.match(r'(?i)^ds_cell_ss\.\d+$',na=False)].copy()
            slots=slots[np.isclose(slots.length_m,52.,atol=1.)].copy()
            slots['s_center_m']=slots.s_m+slots.length_m/2
            slots['ds_from_ip4_m']=slots.s_center_m-ip4
            slots=slots[slots.ds_from_ip4_m.abs()<1000]
            half=self.wiggler_count//2
            if self.wiggler_count%2:
                raise ValueError('Symmetric S4 insertion assumes even number of wigglers')
            minus=slots[slots.ds_from_ip4_m<0].copy()
            plus=slots[slots.ds_from_ip4_m>0].copy()
            minus=minus.iloc[np.argsort(minus.ds_from_ip4_m.abs().to_numpy())[:half]]
            plus=plus.iloc[np.argsort(plus.ds_from_ip4_m.abs().to_numpy())[:half]]
            if len(minus)!=half or len(plus)!=half:
                raise ValueError('Not enough 52-m S4 straight-section cells for provisional wigglers')
            positions=np.sort(np.r_[minus.s_center_m.to_numpy(),plus.s_center_m.to_numpy()])
        else:
            positions=np.asarray(supplied_positions,float)
            if len(positions)!=self.wiggler_count:
                raise ValueError('wiggler_s_positions_m must contain one position per wiggler')
        return pd.DataFrame({'name':[f'WIG.{i+1:02d}' for i in range(len(positions))],
            's_m':positions,'beta_x_m':self.wiggler_beta_m,
            'beta_y_m':self.wiggler_beta_m,
            'status':'provisional S4 / target beta'} )

    def _kicker_locations(self, supplied, beta_x, beta_y):
        if supplied is None:return pd.DataFrame(columns=['s_m','beta_x_m','beta_y_m'])
        ss=np.atleast_1d(np.asarray(supplied,float))
        line=self.optics_table.drop_duplicates('s_m')
        betx=np.interp(ss,line.s_m,line.beta_x_m) if beta_x is None else np.broadcast_to(beta_x,ss.shape)
        bety=np.interp(ss,line.s_m,line.beta_y_m) if beta_y is None else np.broadcast_to(beta_y,ss.shape)
        return pd.DataFrame({'s_m':ss,'beta_x_m':betx,'beta_y_m':bety})

    @property
    def locations(self):
        return {'RF':self.rf_locations.copy(),'BPM':self.bpm_beta_table.copy(),
                'Wiggler':self.wiggler_locations.copy(),'Kicker':self.kicker_locations.copy()}

    def component_impedance(self,family,plane,frequency_hz):
        if plane not in PLANES:raise ValueError(f'plane={plane!r} expected {PLANES}')
        f=np.atleast_1d(np.asarray(frequency_hz,float))
        if family=='RW':
            z=_interp_log_curves(self._rw_log_grid,self._rw_samples[plane],f)
        else:
            source_f,source_z=self._family_tables[family][plane]
            z=interpolated_complex(source_f,source_z,f)*self._family_multiplier(family,plane)
        if np.ndim(frequency_hz)==0:return complex(z[0])
        return z

    def impedance_sum(self,component_name,n_points=5001,fmin=1e3,fmax=50e9,
                      precision_factor=0.1,weight_by_length=False,length_override=None):
        """Backward-compatible facade on top of the *new* PyWIT total.

        No additional length weighting is allowed: all elements are integrated.
        """
        del precision_factor
        if weight_by_length or length_override is not None:
            raise ValueError('HEBModel now stores ring-integrated impedance. '
                             'Extra length-weighting would double-count the source!')
        f=np.linspace(float(fmin),float(fmax),int(n_points))
        z=self.total.get_component(component_name).impedance(f)
        return f,np.asarray(z,complex)

    def impedance_budget(self,frequency_hz=None, *, validate_total=True):
        """Return one row per frequency with every signed component and PyWIT total."""
        f=np.linspace(0.,50e9,5001) if frequency_hz is None else np.asarray(frequency_hz,float)
        if np.any(f>self.rw_fmax_hz):raise ValueError('Budget frequency above computed RW upper limit')
        out={}
        for plane in PLANES:
            curves={name:np.asarray(self.component_impedance(name,plane,f),complex) for name in FAMILIES}
            ztotal=np.asarray(self.total.get_component(SHORTHAND[plane]).impedance(f),complex)
            if validate_total:
                direct=sum(curves.values())
                diff=np.max(np.abs(ztotal-direct))/max(float(np.max(abs(direct))),1e-20)
                if diff>1e-8:
                    raise AssertionError(f'PyWIT model total differs from component sum in {plane}: {diff:.3e}')
            curves['Total']=ztotal
            out[plane]=pd.DataFrame({'frequency_GHz':f/1e9,
                   **{f'{part}_{name}':fun(z) for name,z in curves.items()
                      for part,fun in (('Re',np.real),('Im',np.imag))}})
        return out

    def bunch_wake_04mm(self, *, sigma_z_m=0.4e-3, df_hz=40e6,
                        fmax_hz=10e12, s_window_mm=(-10,200),
                        include_individual=True):
        """Signed 0.4 mm Gaussian bunch wake from **full complex** impedance.

        Other components are zero *outside their supplied numerical band*.
        No CST data, no additional point-charge deconvolution, no fitted factor.
        Returns V/pC (long) and V/pC/mm (dipoles).
        """
        if fmax_hz>self.rw_fmax_hz:raise ValueError('Requested fmax exceeds RW calculated bandwidth')
        n=int(np.rint(fmax_hz/df_hz))
        f=np.arange(n+1,dtype=float)*float(df_hz)
        dt=1/(2*n*df_hz)
        if sigma_z_m<=0:raise ValueError('sigma_z_m must be positive')
        g=np.exp(-0.5*(2*np.pi*f*sigma_z_m/c)**2)
        t=(np.arange(2*n)-n)*dt
        s_mm=t*c*1e3
        mask=(s_mm>=s_window_mm[0])&(s_mm<=s_window_mm[1])
        result={'s_mm':s_mm[mask]}
        for plane in PLANES:
            names=FAMILIES if include_individual else ('Total',)
            curves={name:(np.asarray(self.total.get_component(SHORTHAND[plane]).impedance(f),complex)
                          if name=='Total' else np.asarray(self.component_impedance(name,plane,f),complex))
                    for name in names}
            if 'Total' not in curves:
                curves['Total']=sum(curves.values())
            tbl={}
            for name,z in curves.items():
                spec=z*g if plane=='long' else -1j*z*g
                spec=spec.copy()
                spec[0]=complex(spec[0].real,0)
                spec[-1]=complex(spec[-1].real,0)
                inv=np.fft.fftshift(np.fft.irfft(spec,n=2*n))/dt
                conv=-1e-12 if plane=='long' else 1e-15
                tbl[name]=inv[mask]*conv
            result[plane]=pd.DataFrame({'s_mm':result['s_mm'],**tbl})
        return result

    def save_outputs(self,directory, *, budget_points=5001, wake=True):
        import matplotlib.pyplot as plt
        dst=Path(directory).expanduser();dst.mkdir(parents=True,exist_ok=True)
        self.source_provenance.to_csv(dst/'HEB_V26_2_source_provenance.csv',index=False)
        self.inventory.to_csv(dst/'HEB_V26_2_inventory.csv',index=False)
        for name,tab in self.locations.items():
            if not tab.empty:tab.to_csv(dst/f'HEB_V26_2_locations_{name}.csv',index=False)
        curves=self.impedance_budget(np.linspace(0,50e9,int(budget_points)))
        for pl,tab in curves.items():tab.to_csv(dst/f'HEB_V26_2_budget_{pl}.csv',index=False)
        colors=dict(zip(FAMILIES,['#217188','#805fb7','#b86877','#d26d44','#e5a12e','#7289a5']))
        fig,ax=plt.subplots(3,2,figsize=(15,11),sharex=True,layout='constrained')
        for row,pl in enumerate(PLANES):
            for col,kind in enumerate(('Re','Im')):
                a=ax[row,col];tab=curves[pl]
                for name in FAMILIES:
                    a.plot(tab.frequency_GHz,tab[f'{kind}_{name}'],color=colors[name],lw=1.25,label=name)
                a.plot(tab.frequency_GHz,tab[f'{kind}_Total'],color='black',lw=2,label='Total')
                a.set_ylabel(('Ω' if pl=='long' else 'Ω/m'))
                a.set_title(f'{pl}: {kind}(Z)')
                a.axhline(0,color='gray',lw=.6)
                a.set_yscale('symlog',linthresh=1 if pl=='long' else 1e3)
                a.grid(alpha=.2,which='both')
                if row==2:a.set_xlabel('Frequency [GHz]')
        handles,labels=ax[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncol=7,frameon=False)
        fig.suptitle('FCC-ee HEB V26.2 • native PyWIT impedance budget')
        fig.savefig(dst/'HEB_V26_2_IMPEDANCE_BUDGET.png',dpi=180)
        plt.close(fig)
        if wake:
            wakes=self.bunch_wake_04mm()
            for pl in PLANES:wakes[pl].to_csv(dst/f'HEB_V26_2_wake_0p4mm_{pl}.csv',index=False)
            fig,ax=plt.subplots(3,1,figsize=(10,10),sharex=True,layout='constrained')
            for i,pl in enumerate(PLANES):
                w=wakes[pl]
                for name in FAMILIES:
                    ax[i].plot(w.s_mm,w[name],lw=1,label=name,color=colors[name])
                ax[i].plot(w.s_mm,w['Total'],color='black',lw=2,label='Total')
                ax[i].set_ylabel('V/pC' if pl=='long' else 'V/pC/mm')
                ax[i].set_title(f'{pl}: 0.4 mm Gaussian bunch wake potential')
                ax[i].grid(alpha=.2)
            ax[-1].set_xlabel('s [mm]')
            handles,labels=ax[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside upper center',ncol=7,frameon=False)
            fig.savefig(dst/'HEB_V26_2_0p4mm_WAKE_BUDGET.png',dpi=180)
            plt.close(fig)
            self._export_headtail_ecsv(dst/'HEB_V26_2_0p4mm_PyHEADTAIL.ecsv',wakes)
        return dst

    def _export_headtail_ecsv(self,path,wakes):
        # Time in nanoseconds: s_mm / (1000*c) * 1e9 = s_mm*1e6/c.
        s=wakes['s_mm'];t=s*1e6/c
        arr=np.column_stack([t,wakes['long']['Total'],wakes['x']['Total'],wakes['y']['Total']])
        with Path(path).open('w') as fh:
            fh.write('# %ECSV 1.0\n# ---\n# datatype:\n')
            for k in ('time','longitudinal','dipole_x','dipole_y'):
                fh.write(f'# - {{name: {k}, datatype: float64}}\n')
            fh.write('# meta: !!omap\n')
            fh.write('# - {description: "HEB V26.2; Rq=0.3 um Gold/TL; 0.4 mm bunch"}\n')
            fh.write(f'# - {{kicker_length_m: "{self.kicker_length_m}"}}\n')
            fh.write('# schema: astropy-2.0\n')
            fh.write('time longitudinal dipole_x dipole_y\n')
            np.savetxt(fh,arr,fmt='%.14e')
