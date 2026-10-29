"""Reproduce Morton Table 2 comparisons without changing existing predictions."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT={
 'BNZ':('benzene',-6.32,.37),'BZF':('benzofuran',-8.04,.44),
 'DEN':('indene',-8.31,.48),'I4B':('isobutylbenzene',-7.09,.35),
 'IND':('indole',-11.23,.94),'N4B':('n-butylbenzene',-8.06,.98),
 'PXE':('p-xylene',-6.97,.98),'OXE':('o-xylene',-8.45,.96)}
def run():
 source=json.loads((ROOT/'analysis/mm_ligand_values.json').read_text())
 rows=[]
 for r in source:
  if r['ligand'] not in EXPERIMENT:continue
  name,ex,unc=EXPERIMENT[r['ligand']]
  h_ie=r['mm_steg']-r['ie']
  rows.append(dict(ligand=r['ligand'],name=name,experimental_enthalpy=ex,
    experimental_uncertainty=unc,experimental_temperature_K=302.15,
    calculation_temperature_K=298.15,buffer='50 mM sodium acetate',pH=5.5,
    source_DOI='10.1021/bi00027a006',source_table=2,source_page=8567,
    mm_steg=r['mm_steg'],weie=r['weie'],ie=r['ie'],
    reported_enthalpy_MMPBSA=r['mm_baseline'],error_MMPBSA=r['mm_baseline']-ex,
    inferred_enthalpy_WEIE=r['dh_steg'],error_WEIE=r['dh_steg']-ex,
    inferred_enthalpy_IE_same_G=h_ie,error_IE_same_G=h_ie-ex))
 stats={}
 for method,key in [('MMPBSA','error_MMPBSA'),('WEIE','error_WEIE'),('IE_same_G','error_IE_same_G')]:
  e=np.array([r[key] for r in rows]);stats[method]=dict(N=len(e),
    MSE=float(e.mean()),MAE=float(abs(e).mean()),RMSE=float(np.sqrt((e*e).mean())))
 obj=dict(source='Morton, Baase and Matthews, Biochemistry 1995, 34, 8564-8575, Table 2',
   conditions='302.15 K, 50 mM sodium acetate, pH 5.5; calculations at 298.15 K; no temperature correction',
   rows=rows,statistics=stats)
 (ROOT/'analysis/experimental_enthalpy_comparison.json').write_text(json.dumps(obj,indent=2))
 with (ROOT/'analysis/experimental_enthalpy_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return obj
if __name__=='__main__':print(json.dumps(run()['statistics'],indent=2))
