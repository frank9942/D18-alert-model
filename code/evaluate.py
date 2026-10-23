"""Portable adult D18 evaluation from prepared frozen scores. Never fits a prediction model."""
from runtime import *
import argparse,time

def evaluate(f,cohort,out,repeats=1000):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Evaluation output directory must be empty')
 out.mkdir(parents=True,exist_ok=True);d=kernels.Cohort(f);d.first=kernels.first_mask(f,d.pos);f=f.assign(_pid=d.pi)
 ev={pop:kernels.Evaluator(f[mask]) for pop,mask in [('all_known',f.y.notna()),('H90_unflagged_known',f.y.notna()&~d.hist)]}
 ones=np.ones(d.np,int);meta={'cohort':cohort};points=[dict(**meta,**r) for r in kernels.calculate({'D18':d},ones)];perf=[];bins=[]
 for pop,x in ev.items():
  r,b=x.evaluate(ones);perf.append(dict(cohort=cohort,variant='D18',population=pop,**r))
  bins.extend(dict(cohort=cohort,population=pop,bin=j+1,events=z[0],positive_events=z[1],mean_prediction=z[2],observed_rate=z[3]) for j,z in enumerate(b))
 boot=[];pb=[];rng=np.random.default_rng(2026101401+COHORTS.index(cohort))
 for rep in range(1,repeats+1):
  w=np.bincount(rng.integers(0,d.np,d.np),minlength=d.np)
  boot.extend(dict(**meta,replicate=rep,**r) for r in kernels.calculate({'D18':d},w))
  for pop,x in ev.items():r,_=x.evaluate(w,diagnostics=False);pb.append(dict(**meta,variant='D18',population=pop,replicate=rep,**r))
  if rep%200==0:print('BOOT',cohort,rep,flush=True)
 def csv(n,x):pd.DataFrame(x).to_csv(out/n,index=False,encoding='utf-8-sig')
 csv('clinical_points.csv',points);csv('performance_points.csv',perf);csv('calibration_bins.csv',bins)
 if repeats:
  assert repeats==1000;csv('clinical_bootstrap.csv',boot);csv('performance_bootstrap.csv',pb)
  csv('clinical_intervals.csv',kernels.summarize(pd.DataFrame(points),pd.DataFrame(boot),['cohort','variant','budget_percent','strategy'],kernels.CM))
  csv('performance_intervals.csv',kernels.summarize(pd.DataFrame(perf),pd.DataFrame(pb),['cohort','variant','population'],kernels.PM))
 save(out/'evaluation.json',dict(status='COMPLETE',cohort=cohort,events=len(f),patients=d.np,replicates=repeats,scores_fixed=True,H90_unchanged=True,unknowns_consume_slots=True,intervals='pointwise, no multiplicity adjustment; diagnostic calibration points only',development_warning='Use nested OOF scores for development; never final-model resubstitution scores'))
 print('EVALUATED',cohort,len(f),repeats,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--cohort',choices=COHORTS,required=True);p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,choices=[0,1000],default=1000);a=p.parse_args()
 with threadpool_limits(limits=1):evaluate(load_events(a.input,a.cohort),a.cohort,a.out,a.repeats)
