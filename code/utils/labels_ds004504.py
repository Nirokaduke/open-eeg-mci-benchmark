import re
import pandas as pd
from pathlib import Path

CANON = {
    'ad':'AD','alz':'AD','alzheimer':'AD','alzheimersdisease':'AD','a':'AD',
    'ftd':'FTD','frontotemporal':'FTD','frontotemporaldementia':'FTD','f':'FTD',
    'hc':'HC','cn':'HC','control':'HC','healthy':'HC','c':'HC'
}
CANDIDATE_COLS = ['diagnosis','diagnoses','dx','group','condition','arm','label','phenotype']

def _canon(v):
    if v is None or (isinstance(v,float) and pd.isna(v)): return None
    s=str(v).strip().lower()
    s=re.sub(r"[^a-z]","",s)
    return CANON.get(s)

def load_participants_labels(participants_tsv: str):
    p=Path(participants_tsv)
    if not p.exists(): raise FileNotFoundError(p)
    df=pd.read_csv(p, sep="\t")
    id_col=None
    for c in df.columns:
        if c.lower() in ['participant_id','participant','subject','subject_id','sub','id']:
            id_col=c; break
    if id_col is None: raise ValueError(f"No subject id column in {list(df.columns)}")
    diag_col=None
    for cand in CANDIDATE_COLS:
        for c in df.columns:
            if c.lower()==cand: diag_col=c; break
        if diag_col: break
    if diag_col is None:
        for c in df.columns:
            vals=df[c].astype(str).str.lower().tolist()
            if any(tok in v for v in vals for tok in ['ad','alzheimer','alzheimers','ftd','frontotemporal','hc','control','healthy','cn','group']):
                diag_col=c; break
    if diag_col is None: raise ValueError("Cannot infer diagnosis/group column.")
    labels=df[diag_col].map(_canon)
    if labels.isna().any():
        def fallback(v):
            s=str(v).strip().lower()
            if s in ['a','ad']: return 'AD'
            if s in ['f','ftd']: return 'FTD'
            if s in ['c','hc','cn','control','healthy']: return 'HC'
            return None
        labels=df[diag_col].map(fallback).fillna(labels)
    if labels.isna().any():
        missing=df.loc[labels.isna(), [id_col, diag_col]].head(10)
        raise ValueError(f"Unmapped label values in '{diag_col}'. Examples:\n{missing}")
    df=df.copy(); df['label']=labels.values
    mapping=dict(zip(df[id_col].astype(str), df['label'].astype(str)))
    return df, mapping
