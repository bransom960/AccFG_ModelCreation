import os

import pandas as pd
from rdkit import Chem


def canonical_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def fg_presence_vector(afg, smiles: str, canonical: bool = True) -> dict:
    """Return a binary 0/1 mapping for every loaded functional group in a molecule.

    This intentionally uses the full graph node set, not the pruned final list returned by
    run(..., show_graph=False), so that all functional groups mentioned in the CLI graph
    are counted as present in the spreadsheet.
    """
    if canonical:
        smiles = canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    _, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True, canonical=False)
    present_fgs = set(fg_graph.nodes)
    return {fg_name: int(fg_name in present_fgs) for fg_name in afg.dict_fgs}


def fg_presence_dataframe(afg, smiles_list, canonical: bool = True, output_csv: str = None, append_to_csv: bool = False):
    """
    Build a spreadsheet-like dataframe with one row per molecule and one column per FG.
    The first column is always the input molecule SMILES.
    """
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    records = []
    for smi in smiles_list:
        canon_smi = canonical_smiles(smi) if canonical else smi
        row = {'Molecule': canon_smi}
        row.update(fg_presence_vector(afg, canon_smi, canonical=False))
        records.append(row)

    columns = ['Molecule'] + list(afg.dict_fgs.keys())
    df = pd.DataFrame.from_records(records, columns=columns)
    df = df.fillna(0)

    if output_csv is not None:
        output_csv = os.path.abspath(output_csv)
        os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)

        if append_to_csv and os.path.exists(output_csv):
            try:
                existing_df = pd.read_csv(output_csv)
            except pd.errors.EmptyDataError:
                existing_df = pd.DataFrame(columns=['Molecule'])

            existing_cols = list(existing_df.columns)
            all_cols = list(dict.fromkeys(existing_cols + columns))
            existing_df = existing_df.reindex(columns=all_cols, fill_value=0)
            existing_mols = set(existing_df['Molecule'].astype(str))
            new_df = df[~df['Molecule'].astype(str).isin(existing_mols)]

            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                combined_df = combined_df.reindex(columns=all_cols, fill_value=0)
                combined_df.to_csv(output_csv, index=False)
                return combined_df

            return existing_df.reindex(columns=all_cols, fill_value=0)

        df.to_csv(output_csv, index=False)

    return df


def update_fg_presence_csv(afg, smiles_list, csv_path, canonical: bool = True):
    """Append new molecules to the CSV and avoid duplicate rows."""
    return fg_presence_dataframe(
        afg,
        smiles_list,
        canonical=canonical,
        output_csv=csv_path,
        append_to_csv=True,
    )
