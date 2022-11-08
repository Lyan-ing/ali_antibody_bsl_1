import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import List, Tuple, Union, Optional

Affinity_cols = ['SPR RBD (KD; nm)', 'SPR S1 (KD; nm)', 'SPR S2 (KD; nm)', 'SPR S-ECD (KD; nm)', "SPR S (KD; nm)",
                 'SPR NTD (KD; nm)', 'SPR N (KD; nm)', 'BLI RBD (KD; nm)', 'BLI S1 (KD; nm)',
                 'BLI S (KD; nm)', 'BLI NTD (KD; nm)', 'BLI N (KD; nm)',
                 'MST RBD (KD; nm)', 'ELISA RBD competitive (IC50; μg/ml)',
                 'ELISA S1 competitive (IC50; μg/ml)',
                 'ELISA S competitive (IC50; μg/ml)',
                 'ELISA S competitive (IC80; μg/ml)',
                 'ELISA NTD competitive (IC50; μg/ml)',
                 'ELISA RBD binding (EC50; μg/ml)', 'ELISA S1 binding (EC50; μg/ml)',
                 'ELISA S binding (EC50; μg/ml)', 'ELISA N binding (EC50; μg/ml)',
                 'FACS RBD (IC50; nm/ml)', 'FACS S (IC50; nm/ml)']
Neutralization_cols = ['Live Virus Neutralisation IC50 (50% titre; μg/ml)阈值2μg/ml',
                       'Live Virus Neutralisation IC80 (80% titre; μg/ml)',
                       'Live Virus Neutralisation IC90 (90% titre; μg/ml)',
                       'Live Virus Neutralisation IC100 (100% titre; μg/ml)',
                       'Pseudo Virus Neutralisation IC50 (50% titre; μg/ml)',
                       'Pseudo Virus Neutralisation IC80 (80% titre; μg/ml)',
                       'Pseudo Virus Neutralisation IC90 (90% titre; μg/ml)',
                       'Pseudo Virus Neutralisation IC100 (100% titre; μg/ml)',
                       'Pseudo Virus Neutralisation (fold change)']
antigen_list = ['SARS-CoV1', 'SARS-CoV2_WT', 'Alpha', 'Beta', 'Gamma', 'Delta',
                'Kappa', 'Omicron']


def is_valid(x: str) -> bool:
    left_paren_cnt = right_paren_cnt = 0

    for c in x:
        if c == "(":
            left_paren_cnt += 1
        if c == ")":
            right_paren_cnt += 1
        if left_paren_cnt > 1 or right_paren_cnt > 1:
            return False

    if left_paren_cnt != right_paren_cnt:
        return False

    return True


# function for formatting each Affinity item
def format_affinity(x: str, wt: str) -> Optional[Tuple[List[str], List[str]]]:
    numerics, binds = [], []
    split_ = list(map(lambda y: y.strip(), x.split(";")))

    for s in split_:
        # first ensure "(" and ")" appear at most once
        if not is_valid(s):
            return

        # search for binds
        rex = re.search(r"\(.+\)", s)
        # no parenthesis   remove 括号后，添加种类
        if rex is None:
            numerics.append(s.replace("<", "").strip())
            binds.append(wt)
        else:
            b = rex.group().lstrip("(").rstrip(")").strip()
            if "WT" in b.upper() or "wild" in b.upper():
                binds.append(wt)
            else:
                binds.append(b)
            numerics.append(s.split("(")[0].replace("<", "").strip())

    return numerics, binds


def get_wt(x: Union[str, float]) -> str:
    if not isinstance(x, str):
        return "SARS-CoV2_WT"

    x_upper = x.upper()
    if "MERS-COV" in x_upper:
        return "MERS-CoV_WT"
    elif "SARS-COV2_WT" in x_upper:
        return "SARS-CoV2_WT"
    elif "SARS-COV2" in x_upper and "SARS-COV1" not in x_upper:
        return "SARS-CoV2_WT"
    elif "SARS-COV1" in x_upper and "SARS-COV2" not in x_upper:
        return "SARS-CoV1_WT"


def affinity_filter_lines(_raw):
    _lines_keep, _lines_todo = [], []
    for i in range(len(_raw)):
        keep = True
        for c in Affinity_cols:
            _item = _raw.at[i, c]
            if isinstance(_item, str) and len(_item) > 0 and "&&" in _item:
                keep = False
                break
        if keep:
            _lines_keep.append(i)
        else:
            _lines_todo.append(i)

    _df = _raw.loc[_lines_keep, :]
    return _df, _lines_todo


def neutralization_filter_lines(_raw):
    _lines_keep, _lines_todo = [], []
    for i in range(len(_raw)):
        keep = True
        for c in Neutralization_cols:
            _item = _raw.at[i, c]
            if isinstance(_item, str) and len(_item) > 0 and "&&" in _item:
                keep = False
                break
        if keep:
            _lines_keep.append(i)
        else:
            _lines_todo.append(i)

    _df = _raw.loc[_lines_keep, :]
    return _df, _lines_todo


def affinity_make_output(_raw, _df, _lines_todo):
    output = []
    # traverse each line
    for i in _df.index:
        line = _df.loc[i]  # series

        # determine wildtype
        wt = get_wt(line["Binds to"])
        if wt is None:
            _lines_todo.append(i)
            continue

        # parse SPRs for current line
        valid = True
        parsed = {}
        for c in _df.columns:
            parsed[c] = []

        # loop thru each SPR
        for c in Affinity_cols:
            item = line.loc[c]  # string or NaN
            if not isinstance(item, str):
                continue

            f = format_affinity(item, wt)
            if f is None:
                valid = False
                break

            numerics, binds = f
            for k, b in enumerate(binds):
                # add new bind if not exists
                if b not in parsed["Binds to"]:
                    parsed["Binds to"].append(b)
                    for c_ in Affinity_cols:
                        parsed[c_].append(np.nan)

                n = numerics[k]
                idx = parsed["Binds to"].index(b)
                parsed[c][idx] = n

        if not valid:
            _lines_todo.append(i)
            continue

        # fill in irrelevant columns
        new_lines_cnt = len(parsed["Binds to"])
        for c in _df.columns:
            if c == "Binds to" or c in Affinity_cols:
                continue
            parsed[c] = [line.loc[c]] * new_lines_cnt

        output.append(pd.DataFrame(parsed))
        # save output
    _df_out = pd.concat(output, ignore_index=True)
    _df_todo = _raw.loc[_lines_todo, :]
    return _df_out, _df_todo


def neutralization_make_output(_raw, _df, _lines_todo):
    output = []
    # traverse each line
    for i in _df.index:
        line = _df.loc[i]  # series

        # determine wildtype
        wt = get_wt(line["Neutralising Vs"])
        if wt is None:
            _lines_todo.append(i)
            continue

        # parse SPRs for current line
        valid = True
        parsed = {}
        for c in _df.columns:
            parsed[c] = []

        # loop thru each SPR
        for c in Neutralization_cols:
            item = line.loc[c]  # string or NaN
            if not isinstance(item, str):
                continue

            f = format_affinity(item, wt)
            if f is None:
                valid = False
                break

            numerics, binds = f
            for k, b in enumerate(binds):
                # add new bind if not exists
                if b not in parsed["Neutralising Vs"]:
                    parsed["Neutralising Vs"].append(b)
                    for c_ in Neutralization_cols:
                        parsed[c_].append(np.nan)

                n = numerics[k]
                idx = parsed["Neutralising Vs"].index(b)
                parsed[c][idx] = n

        if not valid:
            _lines_todo.append(i)
            continue

        # fill in irrelevant columns
        new_lines_cnt = len(parsed["Neutralising Vs"])
        for c in _df.columns:
            if c == "Neutralising Vs" or c in Neutralization_cols:
                continue
            parsed[c] = [line.loc[c]] * new_lines_cnt

        output.append(pd.DataFrame(parsed))
        # save output
    _df_out = pd.concat(output, ignore_index=True)
    _df_todo = _raw.loc[_lines_todo, :]
    return _df_out, _df_todo


def affinity_process(_file):
    print('Start processing %s ...' % _file)
    _filename = _file.split('.')[0]
    raw = pd.read_csv("./data/%s.csv" % _filename, sep=',', encoding="gbk")
    # filter lines - remove items with "&&"
    df, lines_todo = affinity_filter_lines(raw)  # filter out that have multi-experiments value() antibody
    # remove reference info in square brackets
    df_affinity = df.loc[:, Affinity_cols]
    df_affinity = df_affinity.replace(to_replace=r"\[.+\]$", value="", regex=True)
    df.loc[:, Affinity_cols] = df_affinity
    df_out, df_todo = affinity_make_output(raw, df, lines_todo)
    df_out_final = df_out.loc[np.isin(df_out['Binds to'], antigen_list)]
    df_todo['Binds to'] = 'SARS-CoV2_WT'
    df_final = pd.concat([df_out_final, df_todo], axis=0)
    df_final.to_csv("./data/%s_new.csv" % _filename, index=False, encoding="gbk")
    print('%s processed!' % _file)


def neutralization_process(_file):
    print('Start processing %s ...' % _file)
    _filename = _file.split('.')[0]
    raw = pd.read_csv("./data/%s.csv" % _filename, sep=',', encoding="gbk")
    # filter lines - remove items with "&&"
    df, lines_todo = neutralization_filter_lines(raw)
    # remove reference info in square brackets
    df_neutralization = df.loc[:, Neutralization_cols]
    df_neutralization = df_neutralization.replace(to_replace=r"\[.+\]$", value="", regex=True)
    df.loc[:, Neutralization_cols] = df_neutralization
    df_out, df_todo = neutralization_make_output(raw, df, lines_todo)
    df_out_final = df_out.loc[np.isin(df_out['Neutralising Vs'], antigen_list)]
    df_todo['Neutralising Vs'] = 'SARS-CoV2_WT'
    df_final = pd.concat([df_out_final, df_todo], axis=0)
    df_final.to_csv("./data/%s_new.csv" % _filename, index=False, encoding="gbk")
    print('%s processed!' % _file)


if __name__ == '__main__':
    affinity_process("Affinity_train.csv")
    affinity_process("Affinity_extraTrainData.csv")
    neutralization_process("Neutralization_train.csv")
