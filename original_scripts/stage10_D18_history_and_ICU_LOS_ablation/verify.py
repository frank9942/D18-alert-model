"""Independent replay of fitted group ablations and clinical/performance counts."""
from common import *
import analyze
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
def models():
 protect();replayed=0;errs=[];out=[]
 for v in VARIANTS:
  f=training_frame(v);cfg,features=config(v);known=f.y.notna();fold=f.fold.astype(float).astype(int).to_numpy();fit=read(S/f'audit/fit_{v}.json');assert not fit['evaluation_used'];assert len(features)==16 and not set(features)&set(REMOVED[v]);assert 't0_specimen' in features
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

def clinical():
 protect();counts=baselinechecks=performancechecks=weightedchecks=0
 pts=pd.read_csv(S/'results/group_ablation_clinical_points.csv');pp=pd.read_csv(S/'results/group_ablation_performance_points.csv')
 oldpt=pd.read_csv(S9/'results/ablation_clinical_points.csv');oldpp=pd.read_csv(S9/'results/ablation_performance_points.csv')
 observed=[]
 for ci,cname in enumerate(COHORTS):
  fs=analyze.datasets(ci,'group_ablation');ref=fs['D18'];dref=small.Cohort(ref);ones=np.ones(dref.np,int)
  for v,f in fs.items():
   for k in ['episode_id','subject_id','t0','prior_crgn_reported_90d','icu','length_of_stay']:
    assert np.array_equal(f[k].astype(str),ref[k].astype(str)),(cname,v,k)
   assert np.allclose(f.y,ref.y,equal_nan=True)
   d=small.Cohort(f);d.first=base.first_mask(f,d.pos);ctx=d.context(ones)
   for pct in BUDGETS:
    quota=(pct*ctx['eligible_events']+99)//100
    for strategy in STRATEGIES:
     row=pts[pts.cohort.eq(cname)&pts.variant.eq(v)&pts.budget_percent.eq(pct)&pts.strategy.eq(strategy)].iloc[0];q=d.evaluate(ctx,strategy,quota)
     for key in ['new_positive_patients','positive_events','known_nonpositive_events','unknown_events','union_positive_patients']:
      assert np.isclose(row[key],q[key],atol=1e-9);counts+=1
     if strategy in d.orders:
      sel=d.choose(ctx,strategy,quota)>0
      assert not (sel&d.hist).any()
      captured=set(f.loc[sel&d.pos,'subject_id']);hist=set(f.loc[d.hist&d.pos,'subject_id'])
      assert row.new_positive_patients==len(captured-hist) and row.first_add==int((sel&d.first).sum());counts+=2
     assert np.isclose(row.positive_events+row.known_nonpositive_events+row.unknown_events,row.added_alerts)
     if v=='D18':
      r=oldpt[oldpt.cohort.eq(cname)&oldpt.variant.eq(v)&oldpt.budget_percent.eq(pct)&oldpt.strategy.eq(strategy)].iloc[0]
      for key in CM:assert np.isclose(row[key],r[key],atol=1e-9);baselinechecks+=1
   for pop,mask in [('all_known',f.y.notna()),('H90_unflagged_known',f.y.notna()&~d.hist)]:
    y=f.loc[mask,'y'].to_numpy(int);p=f.loc[mask,'score'].to_numpy(float)
    vals={'AUROC':roc_auc_score(y,p),'AP':average_precision_score(y,p),'Brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1])}
    row=pp[pp.cohort.eq(cname)&pp.variant.eq(v)&pp.population.eq(pop)].iloc[0]
    for key,value in vals.items():assert np.isclose(row[key],value,atol=1e-12);performancechecks+=1
    if v=='D18':
     r=oldpp[oldpp.cohort.eq(cname)&oldpp.variant.eq(v)&oldpp.population.eq(pop)].iloc[0]
     for key in PM:assert np.isclose(row[key],r[key],atol=1e-12);baselinechecks+=1
  # Independently replay selected bootstrap weights through original reference evaluator.
  bt=pd.read_csv(S/f'private/group_ablation_{cname}_bootstrap_clinical.csv');rng=np.random.default_rng(2026101401+ci)
  assert set(bt.replicate)==set(range(1,1001))
  for rep in range(1,1001):
   w=np.bincount(rng.integers(0,dref.np,dref.np),minlength=dref.np)
   if rep not in [1,500,1000]:continue
   for v,f in fs.items():
    d=small.Cohort(f);ctx=d.context(w)
    for pct in BUDGETS:
     quota=(pct*ctx['eligible_events']+99)//100
     for st in STRATEGIES:
      row=bt[bt.replicate.eq(rep)&bt.variant.eq(v)&bt.budget_percent.eq(pct)&bt.strategy.eq(st)].iloc[0];q=d.evaluate(ctx,st,quota)
      for key in ['new_positive_patients','positive_events','known_nonpositive_events','unknown_events']:
       assert np.isclose(row[key],q[key],rtol=0,atol=1e-8);weightedchecks+=1
  observed.append(dict(cohort=cname,events=len(ref),patients=dref.np,H90_events=int(dref.hist.sum()),unknown=int((dref.y<0).sum()),retained_original_fields=True))
 for typ,keys in [('clinical',['cohort','variant','budget_percent','strategy','metric']),('performance',['cohort','variant','population','metric'])]:
  cur=pd.read_csv(S/f'results/group_ablation_{typ}_intervals.csv');old=pd.read_csv(S9/f'results/ablation_{typ}_intervals.csv')
  a=cur[cur.variant.eq('D18')].set_index(keys);b=old[old.variant.eq('D18')].set_index(keys).loc[a.index]
  for key in ['estimate','low','high']:
   assert np.allclose(a[key],b[key],equal_nan=True,atol=1e-10);baselinechecks+=len(a)
  assert cur.valid_replicates.ge(950).all()
 for cp in (S/'checkpoints').glob('*.json'):
  z=read(cp);assert z['bootstrap']==1000
  for path,h in z['hashes'].items():assert sha(S/path)==h
 for ci,c in enumerate(COHORTS):
  for v in VARIANTS:
   for typ,metrics in [('clinical',CM),('performance',PM)]:
    bt=pd.read_csv(S/f'private/group_ablation_{c}_bootstrap_{typ}.csv')
    a=bt[bt.variant.eq(v+'_minus_D18')];f=bt[bt.variant.eq(v)];r=bt[bt.variant.eq('D18')]
    keys=['replicate','population'] if typ=='performance' else ['replicate','budget_percent','strategy']
    if typ=='clinical':f=f[f.strategy.isin(a.strategy.unique())];r=r[r.strategy.isin(a.strategy.unique())]
    a=a.set_index(keys);f=f.set_index(keys).loc[a.index];r=r.set_index(keys).loc[a.index]
    for k in metrics:
     if k=='reviews_per_new_positive_patient':continue
     assert np.allclose(a[k],f[k]-r[k],equal_nan=True,atol=1e-10)
 protect();save('audit/verification.json',dict(status='PASS',time=now(),point_count_checks=counts,baseline_points_and_interval_checks=baselinechecks,independent_sklearn_performance_checks=performancechecks,weighted_reference_checks=weightedchecks,H90_and_original_comparator_fields_unchanged=True,cohort_checks=observed,all_bootstrap_differences_verified=True,model_verification=read(S/'audit/model_verification.json'),exploratory=True))
 print('GROUP_ABLATION_ALL_VERIFIED',counts,baselinechecks,performancechecks,weightedchecks,flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if len(sys.argv)>1 and sys.argv[1]=='models':models()
  else:clinical()
