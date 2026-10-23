"""Replay fitted models and audit point estimates with independent reference counts."""
from common import *
from analyze import calculate,CM
from workflow import candidate,cooldown,uniform_ties,patient_counts,METRICS
def run():
 protect();checks=0;replayed=0;maxerr=0.;cvchecks=[];refits=[]
 tr=pd.read_csv(R/'数据准备/private/train_adult_matrix.csv',dtype=str,keep_default_na=False)
 for col in CFG['numeric_features']+['y']:tr[col]=pd.to_numeric(tr[col],errors='coerce')
 fd=tr.fold.astype(float).astype(int).to_numpy();known=tr.y.notna().to_numpy()
 for v in VARIANTS:
  fit=read(S/f'audit/fit_{v}.json');assert fit['status']=='PASS' and not fit['evaluation_used']
  for p,h in fit['hashes'].items():assert sha(S/p)==h
  cfg,features=config(v);x=attach_specimens(tr,0,'event_id') if v=='T0_specimen' else tr.copy();mask=known.copy()
  if v=='H90_unflagged':mask&=~tr.prior_crgn_reported_90d.eq('1').to_numpy();assert 'prior_crgn_reported_90d' not in features
  cv=read(S/f'audit/cv_{v}.json')
  for selected in fit['selected']:
   candidates=[r for r in cv if r['outer']==selected['outer']];assert selected['C']==min(candidates,key=lambda r:(round(r['mean_log_loss']/1e-12)*1e-12,r['C']))['C'];checks+=1
  m=joblib.load(S/f'models/{v}.joblib');C=next(r['C'] for r in fit['selected'] if r['outer']=='final');refit=pipeline(v,C);refit.fit(x.loc[mask,features],x.loc[mask,'y'].astype(int));a=m.named_steps['model'];b=refit.named_steps['model'];assert np.array_equal(a.coef_,b.coef_) and np.array_equal(a.intercept_,b.intercept_)
  refits.append(dict(variant=v,exact_coefficient_and_intercept_refit=True,fit_events=int(mask.sum()),features=features))
  for ci,cname in enumerate(COHORTS):
   saved=pd.read_csv(S/f'private/scores_{v}_{cname}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip').set_index('episode_id')
   if ci==0:
    pred=np.empty(len(x))
    for fold in range(5):
     mm=joblib.load(S/f'models/{v}_outer{fold}.joblib');sel=fd==fold;assert not set(tr.loc[mask&~sel,'patient_id'])&set(tr.loc[sel,'patient_id']);pred[sel]=mm.predict_proba(x.loc[sel,features])[:,1]
    ids=tr.event_id.astype(str)
   else:
    f=load_frame(ci) if ci<3 else parquet(EXTFEATURE)
    for col in CFG['categorical_domains']:f[col]=f[col].fillna('MISSING').astype(str).replace({'1.0':'1','0.0':'0','':'MISSING'})
    f.episode_id=f.episode_id.astype(str)
    if v=='T0_specimen':f=attach_specimens(f,ci)
    pred=m.predict_proba(f[features])[:,1];ids=f.episode_id
   err=float(np.max(np.abs(pred-saved.loc[ids,'score'].to_numpy())));assert err<1e-12;maxerr=max(maxerr,err);replayed+=len(pred)
  cvchecks.append(dict(variant=v,selected_C=C,folds_patient_disjoint=True,unknown_excluded_from_fit=True))
 pointchecks=0;baselinechecks=0;workflowchecks=0;cohort_rows=[]
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);ds={'original_LR':small.Cohort(f)}
  for v in VARIANTS:
   q=pd.read_csv(S/f'private/scores_{v}_{cname}.csv',dtype={'episode_id':str},float_precision='round_trip').set_index('episode_id').loc[f.episode_id.astype(str)];ds[v]=small.Cohort(f.assign(score=q.score.to_numpy()))
  pt=pd.read_csv(S/f'results/bounded_clinical_points_{cname}.csv');old=pd.read_csv(R/f'05_完整对照与成人敏感性分析_20261014/results/algorithms_clinical_points.csv');old=old[old.cohort.eq(cname)]
  for name,d in ds.items():
   d.first=base.first_mask(d.f,d.pos);ctx=d.context(np.ones(d.np,int))
   for pct in BUDGETS:
    k=(pct*ctx['eligible_events']+99)//100
    for strategy in STRATEGIES:
     row=pt[pt.variant.eq(name)&pt.budget_percent.eq(pct)&pt.strategy.eq(strategy)].iloc[0];val=d.evaluate(ctx,strategy,k)
     for key in [z for z in CM if not z.startswith('first')]:assert np.isclose(row[key],val[key],atol=1e-10);pointchecks+=1
     if strategy in d.orders:
      sel=d.choose(ctx,strategy,k)>0;captured=set(d.f.loc[sel&d.pos,'subject_id']);hist=set(d.f.loc[d.hist&d.pos,'subject_id']);assert row.new_positive_patients==len(captured-hist);assert row.first_add==int((sel&d.first).sum());pointchecks+=2
   if name=='original_LR':
    for pct in BUDGETS:
     for strategy in STRATEGIES:
      a=pt[pt.variant.eq(name)&pt.budget_percent.eq(pct)&pt.strategy.eq(strategy)].iloc[0];b=old[old.budget_percent.eq(pct)&old.strategy.eq('LR' if strategy=='model' else strategy)].iloc[0]
      for key in [z for z in CM if not z.startswith('first')]:assert np.isclose(a[key],b[key],atol=1e-10);baselinechecks+=1
  # Independent sequential oracle for workflow decisions, without model outcomes.
  d=ds['original_LR'];saved=parquet(S/f'private/workflow_alert_masks_{cname}.parquet');assert np.array_equal(saved.episode_id.astype(str),f.episode_id.astype(str));wp=pd.read_csv(S/f'results/workflow_points_{cname}.csv')
  for cut in read(S/'workflow_thresholds.json')['thresholds']:
   raw=candidate(d,uniform_ties(f),cut)
   for days in [0,1,3,7]:
    key='__'.join(map(str,[cut['strategy'],cut['development_target_percent'],days]));actual=saved[key].to_numpy(bool)
    oracle=np.zeros(len(f),bool)
    last={}
    for j in f.loc[raw].sort_values(['t0','episode_id'],kind='stable').index:
     pid=f.subject_id.iloc[j];t=f.t0.iloc[j]
     if pid not in last or (t-last[pid]).total_seconds()>=days*86400:oracle[j]=True;last[pid]=t
    assert np.array_equal(actual,oracle);row=wp[wp.strategy.eq(cut['strategy'])&wp.development_target_percent.eq(cut['development_target_percent'])&wp.cooldown_days.eq(days)].iloc[0]
    vv=patient_counts(d,actual).sum(axis=0)
    for key,value in zip(METRICS,vv):assert row[key]==value;workflowchecks+=1
  cohort_rows.append(dict(cohort=cname,events=len(f),patients=d.np))
 protect();save('audit/verification.json',dict(status='PASS',time=now(),model_score_replay_events=replayed,max_absolute_score_error=maxerr,final_models_retrained_exactly=refits,cv_choices_verified=cvchecks,development_cv_choice_checks=checks,independent_point_checks=pointchecks,original_baseline_numeric_checks=baselinechecks,workflow_sequential_oracle_count_checks=workflowchecks,cohorts=cohort_rows,original_model_sha256=sha(R/'模型/D17_H3_Adult_v1.joblib')))
 print('VERIFICATION_PASS',replayed,maxerr,pointchecks,baselinechecks,workflowchecks,flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):run()
