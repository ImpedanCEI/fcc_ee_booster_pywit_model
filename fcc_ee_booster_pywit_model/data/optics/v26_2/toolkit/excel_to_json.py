import pandas as pd
import numpy as np
import json

from pathlib import Path

wrk_dir = Path(__file__).parent
# Pointing to file location and specifying the sheet name to convert
excel_data_fragment = pd.read_excel(wrk_dir / '../Booster_parameter_table.xlsx', 
                                    sheet_name='V26_LCC')

# Converting to data frame
df = pd.DataFrame(excel_data_fragment)
a = df.to_numpy(na_value=None)
# thisisjson = df.to_json(orient="records")
# a = json.loads(thisisjson)

operation_modes = ["z", "h", "w", "t"]
title_modes = ["Z", "ZH", "W", "ttbar"]
dd = {"OPERATION_MODES" : operation_modes}
dd2 = {"OPERATION_MODES" : operation_modes}
for mode in operation_modes:
    dd[mode] = dict()
for row in a:
    l = {"Keyword": row[0], "Unit": row[2], "Z": row[3], "ZH": row[4], "W": row[5], "ttbar": row[6], "comment": row[7]}
    key = l["Keyword"]
    if (key is None):
        continue
    unit = l["Unit"]
    comment = l["comment"]
    fac = 1
    if isinstance(unit, str):
        if len(unit) > 1:
            if unit[0] == "G":
                fac = 1e9
            elif unit[0] == "M":
                fac = 1e6
            elif unit[0] == "k":
                fac = 1e3
            elif unit[0] == "m":
                fac = 1e-3
            elif unit[0] == "u":
                fac = 1e-6
            elif unit[0] == "n":
                fac = 1e-9
            elif unit[0] == "p":
                fac = 1e-12
            elif unit[0] == "f":
                fac = 1e-15
        elif unit == "%":
            fac = 1e-2
    elif isinstance(unit, (float, int)):
        fac = unit
    dd2[key] = dict()
    for op, mode in zip(title_modes, operation_modes):
        val = l[op]
        if unit is not None:
            dd2[key]["unit"] = unit
        if comment is not None:
            dd2[key]["comment"] = comment
        if val is None:
            dd[mode][key] = dd["z"][key]
            dd2[key][mode] = dd2[key]["z"]
        elif isinstance(val, str):
            dd[mode][key] = val
            dd2[key][mode] = val
        else:
            dd[mode][key] = val*fac
            dd2[key][mode] = val        

    with open(wrk_dir / '../Booster_parameter_table.json', 'w') as json_file:
        json.dump(dd, json_file, indent=3, sort_keys=True)
    
    with open(wrk_dir / '../Booster_parameter_table_alternative.json', 'w') as json_file:
        json.dump(dd2, json_file, indent=3, sort_keys=True)