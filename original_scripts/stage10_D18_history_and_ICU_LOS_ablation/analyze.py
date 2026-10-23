"""D18 group ablation; paired clustered bootstrap with frozen H90."""
from common import *
def datasets(ci,family):
 assert family=='group_ablation';f=load_frame(ci)
 return {'D18':f,**{n:scored_variant(ci,n) for n in VARIANTS}}
def evaluate_frames(fs,meta,tag,seed,performance=True):
 checkpoint=S/f'checkpoints/{tag}.json'
 if checkpoint.exists():return
 ds={n:small.Cohort(z) for n,z in fs.items()}
 for d in ds.values():d.first=base.first_mask(d.f,d.pos)
 first=next(iter(ds.values()));ones=np.ones(first.np,int);evs={};bins=[]
 if performance:
  for name,f in fs.items():
   z=f.assign(_pid=first.pi)
   for pop,mask in [('all_known',z.y.notna()),('H90_unflagged_known',z.y.notna()&~first.hist)]:
    evs[name,pop]=perf.Evaluator(z[mask])
 def pcalc(pw,rep=None):
  rows=[];vals={}
  for (name,pop),ev in evs.items():
   val,bn=ev.evaluate(pw,diagnostics=rep is None);vals[name,pop]=val;rows.append(dict(variant=name,population=pop,**val))
   if rep is None:
    for j,(n,y,mp,rate) in enumerate(bn):bins.append(dict(**meta,variant=name,population=pop,bin=j+1,events=n,positive_events=y,mean_prediction=mp,observed_fraction=rate))
  ref=next(iter(ds))
  for name,pop in vals:
   if name!=ref:rows.append(dict(variant=name+'_minus_'+ref,population=pop,**{k:vals[name,pop][k]-vals[ref,pop][k] for k in PM}))
  return rows
 pts=fast['calculate'](ds,ones);pp=pcalc(ones);bt=[];pb=[];rng=np.random.default_rng(seed);start=time.monotonic()
 for rep in range(1,1001):
  pw=np.bincount(rng.integers(0,first.np,first.np),minlength=first.np)
  bt.extend(dict(replicate=rep,**row) for row in fast['calculate'](ds,pw))
  pb.extend(dict(replicate=rep,**row) for row in pcalc(pw,rep))
  if rep%200==0:print('D18_BOOT',tag,rep,round(time.monotonic()-start,1),flush=True)
 for a in [pts,pp,bt,pb]:
  for row in a:row.update(meta)
 iv=base.summarize(pd.DataFrame(pts),pd.DataFrame(bt),list(meta)+['variant','budget_percent','strategy'],CM)
 piv=base.summarize(pd.DataFrame(pp),pd.DataFrame(pb),list(meta)+['variant','population'],PM) if performance else []
 for key,data in [('clinical_points',pts),('clinical_intervals',iv),('performance_points',pp),('performance_intervals',piv),('calibration_bins',bins)]:csv(f'results/{tag}_{key}.csv',data)
 csv(f'private/{tag}_bootstrap_clinical.csv',bt)
 if performance:csv(f'private/{tag}_bootstrap_performance.csv',pb)
 save(f'checkpoints/{tag}.json',dict(status='PASS',time=now(),metadata=meta,events=len(first.f),patients=first.np,models=list(ds),bootstrap=1000,performance=performance,code_sha256=sha(Path(__file__)),hashes={str(p.relative_to(S)):sha(p) for p in list((S/'results').glob(tag+'_*.csv'))+list((S/'private').glob(tag+'_*.csv'))}))
def run(family):
 protect();start=time.monotonic()
 for ci,name in enumerate(COHORTS):
  fs=datasets(ci,family);evaluate_frames(fs,dict(cohort=name,family=family),family+'_'+name,2026101401+ci)
 for key in ['clinical_points','clinical_intervals','performance_points','performance_intervals','calibration_bins']:
  csv(f'results/{family}_{key}.csv',pd.concat([pd.read_csv(S/f'results/{family}_{name}_{key}.csv') for name in COHORTS],ignore_index=True))
 protect();save(f'audit/{family}.json',dict(status='PASS',time=now(),seconds=time.monotonic()-start,model_fixed=True,label_only=family=='endpoints',cohorts=COHORTS,bootstrap=1000))

if __name__=='__main__':
 with threadpool_limits(limits=1):run('group_ablation')
