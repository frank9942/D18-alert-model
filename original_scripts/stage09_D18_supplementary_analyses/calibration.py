"""D18 calibration uncertainty; diagnostic fits, no probability recalibration."""
from common import *
def run():
 protect();save('audit/calibration_math.json',perf.verify_math());allpts=[];allivs=[];allbins=[];counts=[]
 old=pd.read_csv(S7/'results/bounded_performance_points.csv')
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);d=small.Cohort(f);f=f.assign(_pid=d.pi);evs={}
  for pop,mask in [('all_known',f.y.notna()),('H90_unflagged_known',f.y.notna()&~d.hist)]:evs[pop]=perf.Evaluator(f[mask])
  pts=[];bins=[];boots=[];bboots=[];ones=np.ones(d.np,int);rng=np.random.default_rng(2026101401+ci);start=time.monotonic()
  for pop,ev in evs.items():
   row,bb=ev.evaluate(ones);pts.append(dict(cohort=cname,population=pop,**row))
   ref=old[old.cohort.eq(cname)&old.variant.eq('T0_specimen')&old.population.eq(pop)].iloc[0]
   for metric in CAL:assert np.isclose(row[metric],ref[metric],atol=1e-10,rtol=0),(cname,pop,metric)
   for j,(n,y,mp,rate) in enumerate(bb):bins.append(dict(cohort=cname,population=pop,bin=j+1,score_lower=ev.edges[j],score_upper=ev.edges[j+1],events=n,positive_events=y,mean_prediction=mp,observed_rate=rate))
  for rep in range(1,1001):
   pw=np.bincount(rng.integers(0,d.np,d.np),minlength=d.np)
   for pop,ev in evs.items():
    row,bb=ev.evaluate(pw,diagnostics=True);boots.append(dict(cohort=cname,population=pop,replicate=rep,**row))
    for j,(n,y,mp,rate) in enumerate(bb):bboots.append(dict(cohort=cname,population=pop,replicate=rep,bin=j+1,events=n,mean_prediction=mp,observed_rate=rate))
   if rep%100==0:print('D18_CALIBRATION',cname,rep,round(time.monotonic()-start,1),flush=True)
  bt=pd.DataFrame(boots);bb=pd.DataFrame(bboots);iv=base.summarize(pd.DataFrame(pts),bt,['cohort','population'],CAL)
  for row in bins:
   z=bb[bb.population.eq(row['population'])&bb.bin.eq(row['bin'])]
   for metric in ['observed_rate','mean_prediction']:
    a=z[metric].to_numpy(float);a=a[np.isfinite(a)];assert len(a)>=950;lo,hi=np.quantile(a,[.025,.975]);row[metric+'_low']=lo;row[metric+'_high']=hi;row['valid_replicates']=len(a)
  assert all(x['valid_replicates']>=950 for x in iv)
  for pop in evs:counts.append(dict(cohort=cname,population=pop,invalid_calibration_replicates=int((~bt[bt.population.eq(pop)].calibration_valid).sum()),replicates=1000))
  for n,rows in [('points',pts),('intervals',iv),('bins',bins)]:csv(f'results/calibration_{n}_{cname}.csv',rows)
  csv(f'private/calibration_bootstrap_{cname}.csv',bt);csv(f'private/calibration_bins_bootstrap_{cname}.csv',bb)
  allpts+=pts;allivs+=iv;allbins+=bins;del boots,bboots,bt,bb;gc.collect()
 for n,rows in [('points',allpts),('intervals',allivs),('bins',allbins),('validity',counts)]:csv(f'results/calibration_{n}.csv',rows)
 protect();save('audit/calibration.json',dict(status='PASS',time=now(),point_values_reproduce_S7=True,replicates=1000,diagnostic_models_fitted_in_each_resample=True,predictive_model_refit=False,probabilities_recalibrated=False,bin_intervals='pointwise patient-bootstrap intervals with original-score bins fixed',validity=counts))
if __name__=='__main__':
 with threadpool_limits(limits=1):run()

