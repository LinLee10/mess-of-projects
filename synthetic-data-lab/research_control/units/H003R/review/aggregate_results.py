"""Aggregate verified H003 and H003R artifacts without model fitting."""
from pathlib import Path
import csv, json, os
import numpy as np

ROOT=Path(os.environ.get('RECOVERED_ROOT','/mnt/data/recovered'))
OUT=Path(os.environ.get('AUDIT_OUTPUT_ROOT','/mnt/data/continuation/results'))
NAMES={17:'h003-seed17',29:'h003r-seed29',41:'h003r-seed41'}
METHODS=['neighbor_raw','neighbor_rank','adaptive_beta']

def moments(values):
    x=np.asarray(values,dtype=float)
    return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),'values':x.tolist()}

def main():
    rows=[];refs=[];geometry_checks=[]
    for seed,name in NAMES.items():
        audit=json.loads((OUT/f'audit_{name}.json').read_text())
        assert audit['status']=='passed' and not audit['final_test_scored']
        refs.append(audit['real_reference_seed17'])
        root=ROOT/name
        assert json.loads((root/'AUDIT.json').read_text())['status']=='passed'
        ctx=json.loads((OUT/f'context_proximity_seed{seed}.json').read_text())
        for method in METHODS:
            checked=audit['results'][name]['methods'][method]
            q=json.loads((root/'run'/method/'QUALITY.json').read_text())
            c=ctx['methods'][method]
            row={'seed':seed,'method':method,'tree_ba':checked['tree']['balanced_accuracy'],
                 'linear_ba':checked['linear']['balanced_accuracy'], 'tree_correct':checked['tree']['correct'],
                 'near_fraction':checked['proximity']['strictly_below_real_q05_fraction'],
                 'rare_class_recall':checked['tree']['class_recall'][3],
                 'copies':q['copy_count'],'unique_rows':q['unique_count'], 'hard_pass':q['hard_pass_fraction'],
                 'numeric_ks':q['mean_numeric_ks'],'conditional_ks':q['class_balanced_conditional_ks'],
                 'query_mean_error':q['mean_query_error'],'query_max_error':q['max_query_error'],
                 'tail_mean_error':q['tail_mean_abs_error'],
                 'pooled_distance_ks':c['pooled']['ks_distance'], 'pooled_distance_median_ratio':c['pooled']['median_ratio'],
                 'conditional_distance_ks':c['eligibility_sensitivity']['100']['weighted_ks'],
                 'conditional_near_fraction':c['eligibility_sensitivity']['100']['below_context_q05_fraction'],
                 'conditional_below_median':c['eligibility_sensitivity']['100']['below_context_median_fraction']}
            row.update({f'class_{i+1}_recall':v for i,v in enumerate(checked['tree']['class_recall'])})
            rows.append(row)
        with np.load(ROOT/'h003-seed17/run/geometry.npz',allow_pickle=False) as ref, np.load(root/'run/geometry.npz',allow_pickle=False) as current:
            same={k:bool(np.array_equal(ref[k],current[k])) for k in ref.files}
            assert all(same.values())
            geometry_checks.append({'seed':seed,'all_fitted_geometry_arrays_equal_to_seed17':True,'arrays':same})
    assert all(x==refs[0] for x in refs)
    with (OUT/'H003R_per_seed.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'status':'verified_development_only','seeds':list(NAMES),'training_rows':348563,'development_rows':116314,
             'final_test_scored':False,'real_reference':refs[0], 'methods':{},'replication_screen':{},'geometry_retraining':geometry_checks,
             'uncertainty_scope':'Generation seeds only; fixed population split, student random state and pilot seeds.'}
    for method in METHODS:
        r=[x for x in rows if x['method']==method]
        summary['methods'][method]={k:moments([x[k] for x in r]) for k in rows[0] if k not in ['seed','method']}
    for seed in [29,41]:
        r={x['method']:x for x in rows if x['seed']==seed};a=r['adaptive_beta']
        outcome={'near_below_both_controls':all(a['near_fraction']<r[m]['near_fraction'] for m in METHODS[:2]),
                 'within_two_tree_ba_points_of_real':abs(a['tree_ba']-refs[0]['balanced_accuracy'])<=.02,
                 'tree_ba_difference_from_real':a['tree_ba']-refs[0]['balanced_accuracy']}
        outcome['passed']=outcome['near_below_both_controls'] and outcome['within_two_tree_ba_points_of_real']
        summary['replication_screen'][str(seed)]=outcome
    summary['all_registered_screens_passed']=all(v['passed'] for v in summary['replication_screen'].values())
    (OUT/'H003R_aggregate.json').write_text(json.dumps(summary,indent=2)+'\n')
    for method in METHODS:
        m=summary['methods'][method]
        print(method,{k: {'mean':m[k]['mean'],'sd':m[k]['sample_sd']} for k in ['tree_ba','linear_ba','tree_correct','near_fraction','rare_class_recall','conditional_below_median','pooled_distance_ks','conditional_distance_ks','copies']})
    print('Screens',summary['replication_screen'])

if __name__=='__main__': main()
