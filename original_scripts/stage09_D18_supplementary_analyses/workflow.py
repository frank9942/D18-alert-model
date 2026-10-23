"""Chronological extra-alert cooldown at development-frozen thresholds."""
from common import *
import hashlib
DAYS=[0,1,3,7]
METRICS=['alerts','positive_events','known_nonpositive_events','unknown_events','selected_positive_patients','new_positive_patients','first_add','reached_patients']
def uniform_ties(f):return np.array([int(hashlib.sha256(('adult_workflow_v1|'+str(e)).encode()).hexdigest()[:13],16) for e in f.episode_id],dtype=np.int64)
def freeze_thresholds():
 f=load_frame(0);d=small.Cohort(f);tie=uniform_ties(f);idx=np.flatnonzero(d.eligible);los=np.nan_to_num(d.los,nan=-np.inf);rows=[]
 for pct in BUDGETS:
  k=(pct*len(idx)+99)//100
  for strategy in ['D18','icu_los']:
   order=idx[np.lexsort((tie[idx],-d.score[idx]))] if strategy=='D18' else idx[np.lexsort((tie[idx],-los[idx],-d.icu[idx].astype(int)))]
   j=order[k-1];row=dict(strategy=strategy,development_target_percent=pct,development_eligible_events=len(idx),development_alerts=k,tie_cutoff=int(tie[j]))
   if strategy=='D18':row['score_cutoff']=float(d.score[j])
   else:row.update(icu_cutoff=int(d.icu[j]),los_cutoff=float(los[j]))
   assert candidate(d,tie,row).sum()==k;rows.append(row)
 save('workflow_thresholds.json',dict(created_utc=now(),selection='development only; nested OOF scores for D18; no outcomes used to choose cutoffs',thresholds=rows,source_sha256=sha(S7/'private/scores_T0_specimen_development_nested.csv')))
 return rows
def candidate(d,tie,row):
 if row['strategy']=='D18':
  cut=row['score_cutoff'];flag=(d.score>cut)|((d.score==cut)&(tie<=row['tie_cutoff']))
 else:
  los=np.nan_to_num(d.los,nan=-np.inf);i=row['icu_cutoff'];l=row['los_cutoff'];flag=(d.icu.astype(int)>i)|((d.icu.astype(int)==i)&((los>l)|((los==l)&(tie<=row['tie_cutoff']))))
 return flag&d.eligible
def cooldown(d,candidates,days):
 if not days:return candidates.copy()
 indices=np.flatnonzero(candidates);times=pd.DatetimeIndex(pd.to_datetime(d.f.t0,utc=True)).as_unit('ns').asi8//1000000000
 order=indices[np.lexsort((d.f.episode_id.to_numpy()[indices],times[indices],d.pi[indices]))]
 emitted=np.zeros(len(d.f),bool);last=np.full(d.np,-10**18,dtype=np.int64);delta=days*86400
 for j in order:
  p=d.pi[j]
  if times[j]-last[p]>=delta:emitted[j]=True;last[p]=times[j]
 return emitted
def patient_counts(d,mask):
 count=lambda sel:np.bincount(d.pi[sel],minlength=d.np)
 pc=count(mask&d.pos);first=base.first_mask(d.f,d.pos)
 return np.column_stack([count(mask),pc,count(mask&d.neg),count(mask&(d.y<0)),pc>0,(pc>0)&~d.base_pos,count(mask&first),count(mask)>0]).astype(float)
def run():
 protect();assert not (S/'audit/workflow.json').exists();rows=freeze_thresholds();allpoints=[];allivs=[];allcompare=[];allcompareivs=[];audit=[]
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);d=small.Cohort(f);ties=uniform_ties(f);mat=[];meta=[];masks={};first=base.first_mask(f,d.pos);first_h90=int((first&d.hist).sum());baseline=int(d.base_pos.sum())
  for r in rows:
   raw=candidate(d,ties,r)
   for days in DAYS:
    mask=cooldown(d,raw,days);assert np.all(mask<=raw) and not mask[d.hist].any();key=(r['strategy'],r['development_target_percent'],days);masks[key]=mask
    mat.append(patient_counts(d,mask));meta.append(dict(cohort=cname,strategy=r['strategy'],development_target_percent=r['development_target_percent'],cooldown_days=days))
  matrix=np.concatenate(mat,axis=1);point=matrix.sum(axis=0).reshape(len(meta),len(METRICS));assert np.allclose(point[:,0],point[:,1:4].sum(axis=1))
  def expanded(values,basepos,firstbase,eligible,nevents):
   out=[]
   for j,m in enumerate(meta):
    v=dict(zip(METRICS,values[j]));v.update(union_positive_patients=basepos+v['new_positive_patients'],first_combined=firstbase+v['first_add'],alerts_per_1000_events=1000*v['alerts']/nevents,realized_extra_percent=100*v['alerts']/eligible,reviews_per_new_positive_patient=v['alerts']/v['new_positive_patients'] if v['new_positive_patients'] else (np.inf if v['alerts'] else np.nan))
    out.append(dict(**m,**v))
   return out
  points=expanded(point,baseline,first_h90,int(d.eligible.sum()),len(f));boots=[];diffboots=[];comparisons=[]
  def differences(values):
   out=[];lookup={(m['strategy'],m['development_target_percent'],m['cooldown_days']):j for j,m in enumerate(meta)}
   for j,m in enumerate(meta):
    if not m['cooldown_days']:continue
    ref=lookup[m['strategy'],m['development_target_percent'],0];out.append(dict(**m,comparison='cooldown_minus_none',**dict(zip(METRICS,values[j]-values[ref]))))
   return out
  comparisons=differences(point);rng=np.random.default_rng(2026101501+ci);hfirst=np.bincount(d.pi[first&d.hist],minlength=d.np);eligible_counts=np.bincount(d.pi[d.eligible],minlength=d.np);event_counts=np.bincount(d.pi,minlength=d.np)
  for rep in range(1,1001):
   pw=np.bincount(rng.integers(0,d.np,d.np),minlength=d.np);v=(pw@matrix).reshape(len(meta),len(METRICS))
   boots.extend(dict(replicate=rep,**r) for r in expanded(v,int(pw[d.base_pos].sum()),float(pw@hfirst),int(pw@eligible_counts),int(pw@event_counts)))
   diffboots.extend(dict(replicate=rep,**r) for r in differences(v))
   if rep%250==0:print('WORKFLOW_BOOT',cname,rep,flush=True)
  keys=['cohort','strategy','development_target_percent','cooldown_days'];extra=['union_positive_patients','first_combined','alerts_per_1000_events','realized_extra_percent','reviews_per_new_positive_patient']
  iv=base.summarize(pd.DataFrame(points),pd.DataFrame(boots),keys,METRICS+extra);div=base.summarize(pd.DataFrame(comparisons),pd.DataFrame(diffboots),keys+['comparison'],METRICS)
  for name,data in [('points',points),('intervals',iv),('changes',comparisons),('change_intervals',div)]:csv(f'results/workflow_{name}_{cname}.csv',data)
  csv(f'private/workflow_bootstrap_{cname}.csv',boots);csv(f'private/workflow_change_bootstrap_{cname}.csv',diffboots)
  z=f[['episode_id','subject_id','t0']].copy()
  for key,mask in masks.items():z['__'.join(map(str,key))]=mask
  writeparquet(f'private/workflow_alert_masks_{cname}.parquet',z)
  allpoints+=points;allivs+=iv;allcompare+=comparisons;allcompareivs+=div
  audit.append(dict(cohort=cname,events=len(f),patients=d.np,H90_alerts=int(d.hist.sum()),H90_positive_patients=baseline,first_H90=first_h90,settings=len(meta),replicates=1000))
  print('WORKFLOW_COMPLETE',cname,flush=True);del matrix,mat,boots,diffboots;gc.collect()
 for n,x in [('points',allpoints),('intervals',allivs),('changes',allcompare),('change_intervals',allcompareivs)]:csv(f'results/workflow_{n}.csv',x)
 protect();save('audit/workflow.json',dict(status='PASS',time=now(),summary=audit,conditional_uncertainty='thresholds and fitted scores fixed; each patient timeline is resampled as a cluster',realized_workload_not_matched=True,no_future_refill=True))
if __name__=='__main__':
 with threadpool_limits(limits=1):run()
