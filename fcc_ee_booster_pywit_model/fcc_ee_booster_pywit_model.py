import os
import re
import yaml
import numpy as np
import pandas as pd

from pathlib import Path
from types import SimpleNamespace
from scipy.constants import physical_constants

from pywit.model import Model
from pywit.interface import (
    _create_iw2d_input_from_dict,
    create_multiple_elements_using_iw2d,
)

from fcc_ee_booster_pywit_model.utils import compute_betas_and_lengths
from fcc_ee_booster_pywit_model.data.machine_layouts.fcc_ee_heb_layout_b1 import layout_dict
from fcc_ee_booster_pywit_model.package_paths import base_dir


# =========================================================
# TFS reader
# =========================================================
def read_tfs_as_twiss_like(tfs_file):
    """
    Read a MAD-X/Xsuite-like TFS file into a lightweight twiss-like object.

    The returned object has attributes such as:
        twiss.name
        twiss.s
        twiss.betx
        twiss.bety
        twiss.summary.length
        twiss.summary.q1
        twiss.summary.q2

    It also stores the full pandas table as:
        twiss._df
    """

    tfs_file = Path(tfs_file)

    if not tfs_file.exists():
        raise FileNotFoundError(f"TFS file does not exist:\n{tfs_file}")

    headers = {}
    lines = tfs_file.read_text().splitlines()

    col_line_idx = None
    type_line_idx = None
    columns = None

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("@"):
            parts = line.split(maxsplit=3)

            if len(parts) >= 4:
                key = parts[1].lower()
                val = parts[3].strip()

                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                else:
                    try:
                        val = float(val)
                    except Exception:
                        pass

                headers[key] = val

        elif line.startswith("*"):
            col_line_idx = i
            columns = line.replace("*", "").split()

        elif line.startswith("$"):
            type_line_idx = i
            break

    if col_line_idx is None or type_line_idx is None or columns is None:
        raise ValueError(f"Cannot find TFS column/type lines in:\n{tfs_file}")

    df = pd.read_csv(
        tfs_file,
        sep=r"\s+",
        names=columns,
        skiprows=type_line_idx + 1,
        engine="python",
    )

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('"', "", regex=False)

    # Normalize dataframe column names to uppercase.
    df.columns = [c.upper() for c in df.columns]

    twiss = SimpleNamespace()

    for col in df.columns:
        setattr(twiss, col.lower(), df[col].to_numpy())

    summary = SimpleNamespace()
    summary.length = float(headers.get("length", np.nan))
    summary.q1 = float(headers.get("q1", np.nan))
    summary.q2 = float(headers.get("q2", np.nan))

    twiss.summary = summary
    twiss._df = df
    twiss._headers = headers

    return twiss


# =========================================================
# IW2D YAML reader
# =========================================================
def create_iw2d_input_from_yaml(name, yaml_file):
    yaml_file = Path(yaml_file)

    if not yaml_file.exists():
        raise FileNotFoundError(f"IW2D YAML file does not exist:\n{yaml_file}")

    with open(yaml_file, "r") as f:
        inputs = yaml.safe_load(f)

    if name not in inputs:
        available = list(inputs.keys())
        raise KeyError(
            f"Name '{name}' is not found in IW2D YAML file:\n"
            f"{yaml_file}\n\n"
            f"Available keys are:\n{available}"
        )

    d = inputs[name]

    return _create_iw2d_input_from_dict(d)


# =========================================================
# Name cleaning helpers
# =========================================================
def _clean_name(name):
    return str(name).strip().replace('"', "")


def _make_name_index(df):
    return {
        _clean_name(name): i
        for i, name in enumerate(df["NAME"].to_numpy(dtype=str))
    }


def _row_to_beta_record(df, idx, name, source="existing"):
    return {
        "name": _clean_name(name),
        "s": float(df.iloc[idx]["S"]),
        "betx": float(df.iloc[idx]["BETX"]),
        "bety": float(df.iloc[idx]["BETY"]),
        "source": source,
    }


# =========================================================
# RF beta weighting
# =========================================================
def find_rf_beta_from_tfs_df(df, rf_name_hint="ac800"):
    """
    Find RF cavity occurrence in the TFS dataframe.

    The new JSON/TFS seems to contain one lumped RF element,
    for example ac800_5. This function finds it using rf_name_hint.
    """

    names = [_clean_name(n) for n in df["NAME"].to_numpy(dtype=str)]

    rf_candidates = [
        name for name in names
        if rf_name_hint.lower() in name.lower()
    ]

    if len(rf_candidates) == 0:
        raise ValueError(
            f"No RF candidate found using rf_name_hint='{rf_name_hint}'.\n"
            "Please check the RF element name in the TFS file."
        )

    if len(rf_candidates) > 1:
        print("Warning: multiple RF candidates found:")
        for name in rf_candidates:
            print("  ", name)
        print("Using the first one.")

    rf_name = rf_candidates[0]
    idx = names.index(rf_name)

    return _row_to_beta_record(df, idx, rf_name, source="existing_rf")


# =========================================================
# BPM beta weighting
# =========================================================
def expected_bpm_names():
    """
    Expected BPM list from your current assumption:

        Arc:
            8 sections, 303 BPMs each

        Straight:
            odd sections s1, s3, s5, s7: 31 BPMs each
            even sections s2, s4, s6, s8: 43 BPMs each

    Total:
        8*303 + 4*31 + 4*43 = 2720
    """

    expected = []

    for a in range(1, 9):
        sec = f"a{a}"
        for n in range(1, 304):
            expected.append((f"bpm.{sec}.{n:03d}", sec, n))

    for s in range(1, 9):
        sec = f"s{s}"
        nmax = 31 if s % 2 == 1 else 43

        for n in range(1, nmax + 1):
            expected.append((f"bpm.{sec}.{n:03d}", sec, n))

    return expected


def estimate_missing_bpm_beta(existing_by_sec_num, sec, num):
    """
    Estimate missing BPM beta.

    Main expected missing BPMs:
        bpm.s1.016
        bpm.s3.016
        bpm.s5.016
        bpm.s7.016

    Priority:
        1. Average same-section neighbors, e.g. 015 and 017.
        2. Average same BPM index over similar sections.
        3. Use nearest BPMs in the same section.
    """

    # 1. Neighbor interpolation in same section
    left = existing_by_sec_num.get((sec, num - 1))
    right = existing_by_sec_num.get((sec, num + 1))

    if left is not None and right is not None:
        return {
            "name": f"bpm.{sec}.{num:03d}",
            "s": np.nan,
            "betx": 0.5 * (left["betx"] + right["betx"]),
            "bety": 0.5 * (left["bety"] + right["bety"]),
            "source": "estimated_from_neighbors",
        }

    # 2. Similar sections
    if sec.startswith("a"):
        similar_secs = [f"a{i}" for i in range(1, 9) if f"a{i}" != sec]
        source_label = "estimated_from_other_arcs"

    elif sec.startswith("s"):
        sidx = int(sec[1:])

        if sidx % 2 == 1:
            similar_secs = [f"s{i}" for i in [1, 3, 5, 7] if f"s{i}" != sec]
            source_label = "estimated_from_other_odd_straights"
        else:
            similar_secs = [f"s{i}" for i in [2, 4, 6, 8] if f"s{i}" != sec]
            source_label = "estimated_from_other_even_straights"

    else:
        similar_secs = []
        source_label = "estimated_from_similar_sections"

    vals = [
        existing_by_sec_num[(other_sec, num)]
        for other_sec in similar_secs
        if (other_sec, num) in existing_by_sec_num
    ]

    if len(vals) > 0:
        return {
            "name": f"bpm.{sec}.{num:03d}",
            "s": np.nan,
            "betx": float(np.mean([v["betx"] for v in vals])),
            "bety": float(np.mean([v["bety"] for v in vals])),
            "source": source_label,
        }

    # 3. Nearest BPMs in the same section
    same_sec = [
        (abs(n - num), rec)
        for (s, n), rec in existing_by_sec_num.items()
        if s == sec
    ]

    if len(same_sec) == 0:
        available_secs = sorted(set(s for s, _ in existing_by_sec_num.keys()))

        raise ValueError(
            f"Cannot estimate missing BPM beta for bpm.{sec}.{num:03d}.\n"
            f"No BPMs were recognized in section {sec}.\n"
            f"Recognized sections are:\n{available_secs}\n\n"
            "This usually means the BPM name parser did not match the TFS NAME column."
        )

    same_sec = sorted(same_sec, key=lambda x: x[0])
    nearest = [rec for _, rec in same_sec[:2]]

    return {
        "name": f"bpm.{sec}.{num:03d}",
        "s": np.nan,
        "betx": float(np.mean([v["betx"] for v in nearest])),
        "bety": float(np.mean([v["bety"] for v in nearest])),
        "source": "estimated_from_nearest_same_section",
    }


def build_bpm_beta_table_from_tfs_df(df, verbose=True):
    """
    Build BPM beta table with expected total 2720 BPMs.

    Robust against:
        uppercase/lowercase names
        quotes around names
        whitespace

    Returns
    -------
    bpm_df : pandas.DataFrame
        Columns:
            name, s, betx, bety, source

    missing : list
        Names that were not present in the TFS and were filled.
    """

    name_to_idx = _make_name_index(df)

    bpm_pattern = re.compile(
        r"^bpm\.([as]\d)\.(\d{3})$",
        re.IGNORECASE,
    )

    existing_by_sec_num = {}

    for name, idx in name_to_idx.items():
        clean = _clean_name(name).lower()
        m = bpm_pattern.match(clean)

        if m is None:
            continue

        sec = m.group(1).lower()
        num = int(m.group(2))

        existing_by_sec_num[(sec, num)] = _row_to_beta_record(
            df,
            idx,
            clean,
            source="existing",
        )

    if verbose:
        print()
        print("Recognized BPMs from TFS:")
        print(f"  total recognized = {len(existing_by_sec_num)}")

        all_secs = [f"a{i}" for i in range(1, 9)] + [f"s{i}" for i in range(1, 9)]

        for sec in all_secs:
            nums = sorted(n for (s, n) in existing_by_sec_num.keys() if s == sec)

            if len(nums) == 0:
                print(f"  {sec}: count=0")
            else:
                missing_in_range = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))

                print(
                    f"  {sec}: count={len(nums)}, "
                    f"range={min(nums):03d}-{max(nums):03d}, "
                    f"missing={missing_in_range}"
                )

    records = []
    missing = []

    for name, sec, num in expected_bpm_names():
        rec = existing_by_sec_num.get((sec, num))

        if rec is None:
            rec = estimate_missing_bpm_beta(existing_by_sec_num, sec, num)
            missing.append(name)

        records.append(rec)

    bpm_df = pd.DataFrame(records)

    if verbose:
        print()
        print("Expected BPM table:")
        print(f"  expected BPM count = {len(bpm_df)}")
        print(f"  existing BPM count = {np.sum(bpm_df['source'] == 'existing')}")
        print(f"  filled BPM count   = {np.sum(bpm_df['source'] != 'existing')}")

        if len(missing) > 0:
            print("  filled BPM names:")
            for name in missing:
                print("   ", name)

    return bpm_df, missing


def beta_weighting_sums_from_bpm_table(bpm_df, beta_x_avg, beta_y_avg):
    """
    Compute beta weighting factors for BPM geometric transverse impedance.

        factor_x = sum_i beta_x_i / <beta_x>
        factor_y = sum_i beta_y_i / <beta_y>
    """

    bpm_factor_x = float(
        np.sum(bpm_df["betx"].to_numpy(dtype=float) / beta_x_avg)
    )

    bpm_factor_y = float(
        np.sum(bpm_df["bety"].to_numpy(dtype=float) / beta_y_avg)
    )

    return bpm_factor_x, bpm_factor_y


# =========================================================
# HEBModel
# =========================================================
# HEBModel V26.2 uses Gold/TL for the circular RW.
from .heb_gold_tl import HEBModel  # HEB V26.2 Gold/TL replacement
