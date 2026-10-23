"""Paired fixed-score evaluation; patient bootstrap with quota reranking."""
from common import *
CM=['new_positive_patients','positive_events','known_nonpositive_events','unknown_events','reviews_per_new_positive_patient','union_positive_patients','first_add','first_combined','first_coverage']
PM=['AUROC','AP','Brier','log_loss']
def context(d,pw):
 w=pw[d.pi];return dict(w=w,pw=pw,eligible_events=int(w[d.eligible].sum()),sizes=[int(w[s].sum()) for s in d.strata])
def choose_prefix(order,cumulative,w,k):
 n=int(np.searchsorted(cumulative,k,side='left'))+1 if k else 0;idx=order[:n];take=w[idx].copy()
 if n:take[-1]=k-(int(cumulative[n-2]) if n>1 else 0)
 assert int(take.sum())==k;return idx,take
def fast_value(d,c,strategy,k,selection=None):
 if selection is not None:
  idx,take=selection;ys=d.y[idx];tp=float(take[ys==1].sum());fp=float(take[ys==0].sum());uk=float(take[ys<0].sum());cap=np.zeros(d.np,int)
  np.maximum.at(cap,d.pi[idx[ys==1]],take[ys==1]);new=float(cap[~d.base_pos].sum())
 else:
  ks=[min(k,c['sizes'][0]),max(0,k-c['sizes'][0])];no=np.ones(d.np);tp=fp=uk=0.
  for g,n,t in zip(d.patient_counts,c['sizes'],ks):
   if t:
    tp+=t*np.dot(c['pw'],g['pos'])/n;fp+=t*np.dot(c['pw'],g['neg'])/n;uk+=t*np.dot(c['pw'],g['unknown'])/n
   no*=small.no_hit(n,t,g['pos'])
  new=float(np.dot(c['pw'][~d.base_pos],(1-no)[~d.base_pos]))
 first_h90=float(c['w'][d.first&d.hist].sum());first_total=float(c['w'][d.first].sum())
 if selection is not None:first_add=float(take[d.first[idx]].sum())
 else:first_add=sum(t*float(c['w'][d.first&st].sum())/n if n else 0 for st,n,t in zip(d.strata,c['sizes'],ks))
 baseline=int(c['pw'][d.base_pos].sum());assert np.isclose(tp+fp+uk,k,atol=1e-8)
 return dict(first_h90=first_h90,first_total=first_total,first_add=first_add,first_combined=first_h90+first_add,first_coverage=(first_h90+first_add)/first_total if first_total else np.nan,added_alerts=k,positive_events=tp,known_nonpositive_events=fp,unknown_events=uk,new_positive_patients=new,baseline_positive_patients=baseline,union_positive_patients=baseline+new,reviews_per_new_positive_patient=k/new if new else np.inf,eligible_events=c['eligible_events'])
def calculate(ds,pw):
 vals={};out=[];first=next(iter(ds.values()));c=context(first,pw);cumulative={}
 for name,d in ds.items():
  assert np.array_equal(d.pi,first.pi) and np.array_equal(d.hist,first.hist)
  cumulative[name]={s:np.cumsum(c['w'][o]) for s,o in d.orders.items()}
  for pct in BUDGETS:
   k=(pct*c['eligible_events']+99)//100
   for strategy in STRATEGIES:
    selection=choose_prefix(d.orders[strategy],cumulative[name][strategy],c['w'],k) if strategy in d.orders else None
    v=fast_value(d,c,strategy,k,selection);vals[name,pct,strategy]=v;out.append(dict(variant=name,budget_percent=pct,strategy=strategy,**v))
   for comp in STRATEGIES[1:]:
    a=vals[name,pct,'model'];b=vals[name,pct,comp]
    out.append(dict(variant=name,budget_percent=pct,strategy='model_minus_'+comp,added_alerts=k,**{key:a[key]-b[key] for key in CM if key!='reviews_per_new_positive_patient'}))
 ref=next(iter(ds))
 for name in list(ds)[1:]:
  for pct in BUDGETS:
   for strategy in ['model','model_minus_icu_expected','model_minus_icu_los']:
    a=next(x for x in out if x['variant']==name and x['budget_percent']==pct and x['strategy']==strategy);b=next(x for x in out if x['variant']==ref and x['budget_percent']==pct and x['strategy']==strategy)
    out.append(dict(variant=name+'_minus_'+ref,budget_percent=pct,strategy=strategy,**{key:a[key]-b[key] for key in CM if key!='reviews_per_new_positive_patient'}))
 return out
def check_math():
 qa=small.verify_math();checks=0
 for ci in [0,3]:
  f=load_frame(ci);d=small.Cohort(f);d.first=base.first_mask(d.f,d.pos);rng=np.random.default_rng(184)
  for pw in [np.ones(d.np,int),np.bincount(rng.integers(0,d.np,d.np),minlength=d.np)]:
   c=d.context(pw);a=calculate({'model':d},pw)
   for r in a:
    if r['strategy'] not in STRATEGIES:continue
    k=(r['budget_percent']*c['eligible_events']+99)//100;b=d.evaluate(c,r['strategy'],k)
    for key in [k for k in CM if not k.startswith('first')]:
     assert np.isclose(r[key],b[key],rtol=0,atol=1e-10),(ci,r['strategy'],key,r[key],b[key]);checks+=1
 save('audit/allocation_math.json',dict(status='PASS',inherited_exhaustive_checks=qa,optimized_vs_reference_checks=checks))
def evaluate(family):
 assert family=='bounded';protect();assert not (S/f'audit/evaluation_{family}.json').exists();start=time.monotonic()
 allpoints=[];allivs=[];allpp=[];allpivs=[];bins=[]
 for ci,cname in enumerate(COHORTS):
  checkpoint=S/f'checkpoints/{family}_{cname}.json'
  if checkpoint.exists():
   for name,dest in [('clinical_points',allpoints),('clinical_intervals',allivs),('performance_points',allpp),('performance_intervals',allpivs)]:dest.extend(pd.read_csv(S/f'results/{family}_{name}_{cname}.csv').to_dict('records'))
   bins.extend(read(checkpoint).get('calibration_bins',[]));continue
  f=load_frame(ci);keep=['episode_id','subject_id','t0','y','icu','tie','score','length_of_stay','prior_crgn_reported_90d'];f=f[keep];fs={}
  if family=='endpoints':
   ep=parquet(S/f'private/endpoints_{cname}.parquet');assert np.array_equal(ep.episode_id.astype(str),f.episode_id.astype(str))
   for name in ENDPOINTS:fs[name]=f.assign(y=ep[name].to_numpy(float))
  else:
   fs['original_LR']=f
   for name in VARIANTS:
    sc=pd.read_csv(S/f'private/scores_{name}_{cname}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip');assert sc.episode_id.is_unique;sc=sc.set_index('episode_id').loc[f.episode_id.astype(str)]
    if 'subject_id' in sc:assert np.array_equal(sc.subject_id.astype(str),f.subject_id.astype(str))
    fs[name]=f.assign(score=sc.score.to_numpy(float))
  ds={n:small.Cohort(z) for n,z in fs.items()}
  for d in ds.values():d.first=base.first_mask(d.f,d.pos)
  d=next(iter(ds.values()));evs={};localbins=[]
  for name,z in fs.items():
   zz=z.assign(_pid=d.pi);pops=[('H90_unflagged_known',zz.y.notna()&~d.hist)]
   if name!='H90_unflagged':pops.append(('all_known',zz.y.notna()))
   for pop,mask in pops:evs[name,pop]=perf.Evaluator(zz[mask])
  def performance(pw,rep=None):
   rows=[];vv={}
   for (name,pop),ev in evs.items():
    v,cal=ev.evaluate(pw,diagnostics=rep is None);vv[name,pop]=v;rows.append(dict(variant=name,population=pop,**v))
    if rep is None:
     for j,(n,y,p,o) in enumerate(cal):localbins.append(dict(cohort=cname,variant=name,population=pop,bin=j+1,events=n,positive_events=y,mean_prediction=p,observed_fraction=o))
   ref=next(iter(ds))
   for name,pop in vv:
    if name==ref:continue
    rows.append(dict(variant=name+'_minus_'+ref,population=pop,**{k:vv[name,pop][k]-vv[ref,pop][k] for k in PM}))
   return rows
  ones=np.ones(d.np,int);points=calculate(ds,ones);pp=performance(ones);boots=[];pbs=[];rng=np.random.default_rng(2026101401+ci)
  print('POINTS',family,cname,[(r['variant'],r['new_positive_patients']) for r in points if r['budget_percent']==2 and r['strategy']=='model'],flush=True)
  for rep in range(1,1001):
   pw=np.bincount(rng.integers(0,d.np,d.np),minlength=d.np)
   boots.extend(dict(replicate=rep,**r) for r in calculate(ds,pw));pbs.extend(dict(replicate=rep,**r) for r in performance(pw,rep))
   if rep%200==0:print('BOOT',family,cname,rep,'seconds',round(time.monotonic()-start,1),flush=True)
  for data in [points,pp,boots,pbs]:
   for row in data:row['cohort']=cname
  pt=pd.DataFrame(points);bt=pd.DataFrame(boots);pf=pd.DataFrame(pp);pb=pd.DataFrame(pbs)
  iv=base.summarize(pt,bt,['cohort','variant','budget_percent','strategy'],CM);piv=base.summarize(pf,pb,['cohort','variant','population'],PM)
  for name,data,dest in [('clinical_points',points,allpoints),('clinical_intervals',iv,allivs),('performance_points',pp,allpp),('performance_intervals',piv,allpivs)]:csv(f'results/{family}_{name}_{cname}.csv',data);dest.extend(data)
  csv(f'private/bootstrap_{family}_clinical_{cname}.csv',bt);csv(f'private/bootstrap_{family}_performance_{cname}.csv',pb);bins.extend(localbins)
  save(f'checkpoints/{family}_{cname}.json',dict(status='PASS',time=now(),replicates=1000,calibration_bins=localbins))
  del fs,ds,evs,boots,pbs,bt,pb;gc.collect()
 for name,data in [('clinical_points',allpoints),('clinical_intervals',allivs),('performance_points',allpp),('performance_intervals',allpivs),('calibration_bins',bins)]:csv(f'results/{family}_{name}.csv',data)
 protect();save(f'audit/evaluation_{family}.json',dict(status='PASS',time=now(),seconds=time.monotonic()-start,replicates_per_cohort=1000,cohorts=COHORTS,models_fixed=True,endpoint_sensitivity_changes_labels_only=family=='endpoints',calibration='point diagnostic only; no recalibration'))
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if sys.argv[1]=='verify-math':check_math()
  else:evaluate(sys.argv[1])
