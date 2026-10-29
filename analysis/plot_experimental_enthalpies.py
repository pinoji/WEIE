from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'analysis/experimental_enthalpy_comparison.json').read_text())['rows']
y=np.arange(len(r));exp=np.array([x['experimental_enthalpy'] for x in r])
weie=np.array([x['inferred_enthalpy_WEIE'] for x in r]);ie=np.array([x['inferred_enthalpy_IE_same_G'] for x in r])
mm=np.array([x['reported_enthalpy_MMPBSA'] for x in r])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,(a,b)=plt.subplots(1,2,figsize=(10,4.8),sharey=True,layout='constrained')
a.errorbar(exp,y,xerr=[x['experimental_uncertainty'] for x in r],fmt='o',color='#263746',capsize=3,label='Experiment (302.15 K)',ms=5)
a.scatter(weie,y-.16,marker='s',color='#18814e',s=30,label='MM-STEG/WEIE')
a.scatter(ie,y+.16,marker='^',color='#c87713',s=32,label='MM-STEG/IE')
a.scatter(mm,y+.30,marker='D',color='#7d5091',s=25,label='Conventional MM/PBSA (L)')
a.set_yticks(y,[x['ligand'] for x in r]);a.invert_yaxis()
a.set_xlabel(r'Binding enthalpy ($\mathrm{kcal}\,\mathrm{mol}^{-1}$)');a.set_title('(a) Experimental and inferred enthalpies',loc='left',fontsize=10)
fig.legend(*a.get_legend_handles_labels(), loc='outside lower center', ncol=2, fontsize=8)
b.axvline(0,color='#263746',lw=.8);b.scatter(weie-exp,y-.16,marker='s',color='#18814e',s=30);b.scatter(ie-exp,y+.16,marker='^',color='#c87713',s=32)
b.scatter(mm-exp,y+.30,marker='D',color='#7d5091',s=25)
for i in y:b.plot([weie[i]-exp[i],ie[i]-exp[i]],[i-.16,i+.16],color='.75',lw=.8,zorder=0)
b.set_xlabel(r'Signed enthalpy error ($\mathrm{kcal}\,\mathrm{mol}^{-1}$)');b.set_title('(b) Estimate minus experiment',loc='left',fontsize=10)
for ax in [a,b]:ax.grid(axis='x',alpha=.15)
fig.savefig(ROOT/'figures/fig_t4_enthalpy_comparison.pdf',bbox_inches='tight')
