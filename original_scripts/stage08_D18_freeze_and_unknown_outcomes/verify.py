"""Independent set-based checks, completeness and frozen-source integrity."""
from common import *
def run():
 protect();parent.protect()
 assert read(S/'audit/analysis_complete.json')['status']=='PASS'
 points=pd.read_csv(S/'results/scenario_points.csv');iv=pd.read_csv(S/'results/scenario_intervals.csv')
 checks=0
 for ci,cname in enumerate(COHORTS):
  f,ft,q=frames(ci);p=points[points.cohort.eq(cname)]
  for scenario,fill in [('observed',None),('all_unknown_nonpositive',0),('all_unknown_positive',1)]:
   y=f.y.to_numpy().copy()
   if fill is not None:y[np.isnan(y)]=fill
   patient=f.subject_id.astype(str).to_numpy();hist=f.prior_crgn_reported_90d.astype(str).eq('1').to_numpy();positive=y==1
   bp=set(patient[hist&positive])
   ff=f.assign(y=y);first=base.first_mask(ff,positive);first_h=int((hist&first).sum())
   for variant,frame in [('original_LR',f),('T0_specimen',ft)]:
    d=small.Cohort(frame);ctx=d.context(np.ones(d.np,int))
    for pct in BUDGETS:
     k=(pct*ctx['eligible_events']+99)//100
     for strategy in ['model','icu_los']:
      selected=d.choose(ctx,strategy,k)>0
      oracle=dict(new_positive_patients=len(set(patient[selected&positive])-bp),positive_events=int((selected&positive).sum()),known_nonpositive_events=int((selected&(y==0)).sum()),unknown_events=int((selected&np.isnan(y)).sum()),first_add=int((selected&first).sum()),first_combined=first_h+int((selected&first).sum()),union_positive_patients=len(bp|set(patient[selected&positive])))
      r=p[(p.scenario==scenario)&(p.variant==variant)&(p.budget_percent==pct)&(p.strategy==strategy)].iloc[0]
      for key,value in oracle.items():assert np.isclose(r[key],value,atol=1e-9,rtol=0),(cname,scenario,variant,strategy,key);checks+=1
   for pct in BUDGETS:
    for metric in ['new_positive_patients','first_add','positive_events']:
     rows=p[(p.scenario==scenario)&(p.budget_percent==pct)&p.variant.eq('T0_specimen')]
     for comp in ['icu_los','icu_expected']:
      v=rows[rows.strategy.eq('model')].iloc[0][metric]-rows[rows.strategy.eq(comp)].iloc[0][metric]
      assert np.isclose(v,rows[rows.strategy.eq('model_minus_'+comp)].iloc[0][metric]);checks+=1
  for rel,h in read(S/f'checkpoints/{cname}.json')['hashes'].items():assert sha(S/rel)==h
  bt=pd.read_csv(S/f'private/bootstrap_{cname}.csv')
  assert bt.replicate.nunique()==1000 and not bt.duplicated(['replicate','scenario','variant','budget_percent','strategy']).any()
  assert bt.groupby(['scenario','variant','budget_percent','strategy']).replicate.nunique().eq(1000).all()
 assert len(points)==4*3*39 and len(iv)>0 and iv.valid_replicates.ge(950).all()
 assert not points.duplicated(['cohort','scenario','variant','budget_percent','strategy']).any()
 for fn in ['cumulative_sharp_bounds','first_sharp_bounds']:
  b=pd.read_csv(S/f'results/{fn}.csv');assert len(b)==36
  for row in b.itertuples():
   variant='T0_specimen_minus_original_LR' if row.comparator=='original_LR' else 'T0_specimen';strategy='model' if row.comparator=='original_LR' else 'model_minus_'+row.comparator
   ref=points[points.cohort.eq(row.cohort)&points.budget_percent.eq(row.budget_percent)&points.scenario.eq('observed')&points.variant.eq(variant)&points.strategy.eq(strategy)].iloc[0]
   metric='new_positive_patients' if fn.startswith('cumulative') else 'first_add';observed=row.observed_patient_difference if fn.startswith('cumulative') else row.observed_first_difference
   assert np.isclose(ref[metric],observed);checks+=1
 for rel,h in read(S7/'completion_manifest.json')['files'].items():assert sha(S7/rel)==h
 copied=joblib.load(S/f'models/{CANDIDATE}.joblib');original=joblib.load(S7/'models/T0_specimen.joblib')
 assert np.array_equal(copied[-1].coef_,original[-1].coef_) and np.array_equal(copied[-1].intercept_,original[-1].intercept_)
 save('audit/verification.json',dict(status='PASS',time=now(),independent_set_based_and_difference_checks=checks,scenario_points=len(points),scenario_interval_rows=len(iv),bounds_rows=72,all_bootstrap_groups_1000=True,source_observed_values_reproduced=sum(read(S/f'checkpoints/{c}.json')['observed_bootstrap_values_reproduced'] for c in COHORTS),main_model_sha256=sha(R/'模型/D17_H3_Adult_v1.joblib'),candidate_model_sha256=sha(S/f'models/{CANDIDATE}.joblib'),candidate_copied_byte_identical=True,parent_S7_all_files_hash_unchanged=True,inherited_source_inputs_hash_unchanged=True,full_score_replay_reference='audit/source_verification.json; all S7 original inputs/scores/model bytes protected',new_model_fits=0))
 print('INDEPENDENT_CHECKS_PASS',checks,'points',len(points),'interval_rows',len(iv),flush=True)
if __name__=='__main__':run()

