"""Independent D18 score replay, refit, selection counts and workflow oracle."""
from common import *
import analyze,workflow
def models():
 protect();replayed=0;errs=[];out=[]
 for v in VARIANTS:
  f=training_frame(v);cfg,features=config(v);known=f.y.notna();fold=f.fold.astype(float).astype(int).to_numpy();fit=read(S/f'audit/fit_{v}.json');assert not fit['evaluation_used']
  for p,h in fit['hashes'].items():assert sha(S/p)==h
  cv=read(S/f'audit/cv_{v}.json')
  for ch in fit['selected']:
   rr=[x for x in cv if x['outer']==ch['outer']];assert ch['C']==min(rr,key=lambda x:(round(x['mean_log_loss']/1e-12)*1e-12,x['C']))['C']
  C=next(x['C'] for x in fit['selected'] if x['outer']=='final');m=joblib.load(S/f'models/{v}.joblib');r=pipeline(v,C);r.fit(f.loc[known,features],f.loc[known,'y'].astype(int))
  assert np.array_equal(m[-1].coef_,r[-1].coef_) and np.array_equal(m[-1].intercept_,r[-1].intercept_)
  for ci,cname in enumerate(COHORTS):
   saved=pd.read_csv(S/f'private/scores_{v}_{cname}.csv',dtype={'episode_id':str},float_precision='round_trip').set_index('episode_id')
   if ci==0:
    pred=np.empty(len(f));ids=f.event_id.astype(str)
    for k in range(5):
     take=fold==k;assert not set(f.loc[known&~take,'patient_id'])&set(f.loc[take,'patient_id']);mm=joblib.load(S/f'models/{v}_outer{k}.joblib');pred[take]=mm.predict_proba(f.loc[take,features])[:,1]
   elif ci<3:
    z=transform(load_frame(ci),v);pred=m.predict_proba(z[features])[:,1];ids=z.episode_id.astype(str)
   else:
    chunks=[];ident=[]
    with duckdb.connect(config={'memory_limit':'256MB','threads':1}) as con:
     cur=con.execute('SELECT * FROM read_parquet(?)',[str(EXTFEATURE)])
     while True:
      z=cur.fetch_df_chunk(16)
      if z.empty:break
      for col in parent.CFG['categorical_domains']:z[col]=z[col].fillna('MISSING').astype(str).replace({'1.0':'1','0.0':'0','':'MISSING'})
      z=transform(parent.attach_specimens(z,ci),v);chunks.append(m.predict_proba(z[features])[:,1]);ident.extend(z.episode_id.astype(str))
    pred=np.concatenate(chunks);ids=ident
   err=float(np.max(abs(pred-saved.loc[ids,'score'].to_numpy())));assert err<1e-12;(errs.append(err));replayed+=len(pred)
  out.append(dict(variant=v,features=features,n_features=len(features),C=C,fit_events=int(known.sum()),retained_development_events=len(f),exact_final_refit=True,outer_fold_replay=True))
 save('audit/model_verification.json',dict(status='PASS',time=now(),models=out,score_replay_events=replayed,max_score_error=max(errs),main_D18_unchanged=sha(MAIN),D17_unchanged=sha(R/'模型/D17_H3_Adult_v1.joblib')))
 print('MODELS_REPLAY_VERIFIED',replayed,flush=True)
def workflow_toy():
 times=pd.to_datetime(['2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-09'])
 f=pd.DataFrame(dict(episode_id=np.arange(5),subject_id=['x']*5,t0=times,y=[1,0,np.nan,1,0],icu=True,tie=np.arange(5),score=.5,prior_crgn_reported_90d='MISSING',length_of_stay=np.arange(5)))
 d=small.Cohort(f);raw=np.ones(5,bool);want=np.array([True,False,False,True,True]);assert np.array_equal(workflow.cooldown(d,raw,3),want)
 for unit in ['us','ns']:
  z=f.copy();z.t0=z.t0.dt.as_unit(unit);assert np.array_equal(workflow.cooldown(small.Cohort(z),raw,3),want)
 z=f.assign(y=0);assert np.array_equal(workflow.cooldown(small.Cohort(z),raw,3),want)
 a=f.iloc[:4].copy();assert np.array_equal(workflow.cooldown(small.Cohort(a),raw[:4],3),want[:4])
 assert np.array_equal(workflow.cooldown(d,raw,0),raw)
 return dict(status='PASS',exact_boundary=True,suppressed_does_not_reset=True,time_units_us_ns=True,outcome_independent=True,future_append_invariant=True)
def clinical():
 protect();checks=0;baselinechecks=0;old=pd.read_csv(S7/'results/bounded_clinical_points.csv')
 for family in ['endpoints','ablation','after90']:
  assert read(S/f'audit/{family}.json')['status']=='PASS'
  pts=pd.read_csv(S/f'results/{family}_clinical_points.csv')
  for ci,cname in enumerate(COHORTS):
   fs=analyze.datasets(ci,family)
   for name,f in fs.items():
    d=small.Cohort(f);d.first=base.first_mask(f,d.pos);ctx=d.context(np.ones(d.np,int))
    for pct in BUDGETS:
     k=(pct*ctx['eligible_events']+99)//100
     for strategy in STRATEGIES:
      row=pts[pts.cohort.eq(cname)&pts.variant.eq(name)&pts.budget_percent.eq(pct)&pts.strategy.eq(strategy)].iloc[0];oracle=d.evaluate(ctx,strategy,k)
      for key in ['new_positive_patients','positive_events','known_nonpositive_events','unknown_events','union_positive_patients']:
       assert np.isclose(row[key],oracle[key],atol=1e-9);checks+=1
      if strategy in d.orders:
       sel=d.choose(ctx,strategy,k)>0;h=set(f.loc[d.hist&d.pos,'subject_id']);capt=set(f.loc[sel&d.pos,'subject_id'])
       assert row.new_positive_patients==len(capt-h) and row.first_add==int((sel&d.first).sum());checks+=2
      if (family=='endpoints' and name=='full_24h') or (family=='ablation' and name=='D18'):
       ref=old[old.cohort.eq(cname)&old.variant.eq('T0_specimen')&old.budget_percent.eq(pct)&old.strategy.eq(strategy)].iloc[0]
       for key in CM:assert np.isclose(row[key],ref[key],atol=1e-9);baselinechecks+=1
  if family=='endpoints':
   for ci,cname in enumerate(COHORTS):
    fs=analyze.datasets(ci,family);assert all(np.array_equal(z.score,fs['full_24h'].score) for z in fs.values());assert all(np.array_equal(z.prior_crgn_reported_90d,fs['full_24h'].prior_crgn_reported_90d) for z in fs.values())
 workflowchecks=0;wold=pd.read_csv(S7/'results/workflow_points.csv')
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);d=small.Cohort(f);saved=parquet(S/f'private/workflow_alert_masks_{cname}.parquet');wp=pd.read_csv(S/f'results/workflow_points_{cname}.csv');assert np.array_equal(saved.episode_id.astype(str),f.episode_id.astype(str))
  for cut in read(S/'workflow_thresholds.json')['thresholds']:
   raw=workflow.candidate(d,workflow.uniform_ties(f),cut)
   for days in workflow.DAYS:
    name='__'.join(map(str,[cut['strategy'],cut['development_target_percent'],days]));actual=saved[name].to_numpy(bool);oracle=np.zeros(len(f),bool);last={}
    for j in f.loc[raw].sort_values(['t0','episode_id'],kind='stable').index:
     pid=f.subject_id.iloc[j];t=f.t0.iloc[j]
     if pid not in last or (t-last[pid]).total_seconds()>=days*86400:oracle[j]=True;last[pid]=t
    assert np.array_equal(actual,oracle)
    row=wp[wp.strategy.eq(cut['strategy'])&wp.development_target_percent.eq(cut['development_target_percent'])&wp.cooldown_days.eq(days)].iloc[0]
    for k,v in zip(workflow.METRICS,workflow.patient_counts(d,actual).sum(axis=0)):assert row[k]==v;workflowchecks+=1
    if cut['strategy']=='icu_los':
     ref=wold[wold.cohort.eq(cname)&wold.strategy.eq('icu_los')&wold.development_target_percent.eq(cut['development_target_percent'])&wold.cooldown_days.eq(days)].iloc[0]
     for k in workflow.METRICS:assert row[k]==ref[k];baselinechecks+=1
 # Hash and bootstrap completeness for all clinical groups.
 for cp in (S/'checkpoints').glob('*.json'):
  q=read(cp);assert q['bootstrap']==1000
  for p,h in q['hashes'].items():assert sha(S/p)==h
 for cp in ['calibration','pathogens','workflow','history']:assert read(S/f'audit/{cp}.json')['status']=='PASS'
 cal=pd.read_csv(S/'results/calibration_intervals.csv');assert cal.valid_replicates.ge(950).all()
 pv=pd.read_csv(S/'results/pathogen_patient_exclusive.csv')
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);d=small.Cohort(f);ctx=d.context(np.ones(d.np,int))
  for pct in BUDGETS:
   ref=d.evaluate(ctx,'model',(pct*ctx['eligible_events']+99)//100)['new_positive_patients'];z=pv[pv.cohort.eq(cname)&pv.group.eq('D18_added')&pv.budget_percent.eq(pct)];assert z.patients.sum()==ref
 protect();save('audit/verification.json',dict(status='PASS',time=now(),independent_clinical_checks=checks,baseline_and_rule_checks=baselinechecks,workflow_oracle_checks=workflowchecks,workflow_toy=workflow_toy(),models=read(S/'audit/model_verification.json'),bootstrap_groups_complete=True,all_six_analyses_verified=True))
 print('D18_ALL_VERIFIED',checks,baselinechecks,workflowchecks,flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if len(sys.argv)>1 and sys.argv[1]=='models':models()
  else:clinical()

