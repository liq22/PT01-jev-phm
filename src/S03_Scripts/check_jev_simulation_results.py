"""Recompute saved prediction statistics and verify trial/selection consistency."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'S01_Package'))
from jev_simulation import per_unit

def main(run):
    run=Path(run);cfg=json.loads((run/'config.json').read_text());df=pd.read_csv(run/'metrics.csv');meta=pd.read_csv(run/'virtual_units.csv')
    count=0;maximum=0.;differences=[]
    for path in run.glob('predictions_*.npz'):
        seed=int(path.stem.split('_')[1]);condition='_'.join(path.stem.split('_')[2:]);z=np.load(path,allow_pickle=False)
        assert len(set(z['unit']))==90 and z['y'].shape==(360,)
        source=np.load(run/f'source_inputs_{seed}.npz',allow_pickle=False)
        su=set().union(*(set(source[k]) for k in source.files if k.endswith('_unit')))
        assert not su&set(z['unit'])
        for method in ('fine_only','joint_event','weighted_libra','conditional_anchor','conditional_freeze','rf_fine'):
            rr=per_unit(z[method],z['y'],z['unit'],z['fine_only']);row=df[(df.replicate==seed)&(df.condition==condition)&(df.method==method)].iloc[0]
            for key in ('nll','coarse_nll','conditional_nll','accuracy','brier','logodds_drift'):
                e=abs(np.mean([r[key] for r in rr])-row[key]);maximum=max(maximum,e);assert e<1e-10
            assert all(abs(r['nll']-r['coarse_nll']-r['conditional_nll'])<1e-12 for r in rr)
            count+=1
        assert all(np.array_equal(z['fine_only'],z[m]) for m in ('joint_event','weighted_libra','conditional_anchor'))
        differences.append(float(np.max(abs(np.log(z['conditional_freeze'][:,1]/z['conditional_freeze'][:,2])-np.log(z['fine_only'][:,1]/z['fine_only'][:,2])))))
    assert count==108
    assert meta.unit.nunique()==1134 and len(meta)==1134
    result={'prediction_metric_rows_checked':count,'max_metric_recomputation_difference':maximum,'source_target_unit_disjoint':True,'virtual_units_total':1134,'virtual_target_units':270,'paired_views_not_independent_units':True,'fine_equals_selected_event_libra_anchor':True,'max_hard_freeze_logodds_difference':max(differences),'nll_chain_rule':True,'scope':'Saved simulated predictions only. Not real-machine validation.'}
    (run/'recomputation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
