"""Offline chronological simulation, using development-frozen workload thresholds."""
from runtime import *
import argparse

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--cohort',choices=COHORTS,required=True);p.add_argument('--out',required=True);p.add_argument('--budget',type=int,choices=[1,2,5],default=2);p.add_argument('--days',type=int,choices=[0,1,3,7],default=3);p.add_argument('--strategy',choices=['D18','icu_los'],default='D18');a=p.parse_args()
 out=Path(a.out)
 if out.exists():raise FileExistsError('Use a new output filename')
 f=load_events(a.input,a.cohort,labels_required=False);d=kernels.Cohort(f)
 row=next(r for r in read(ROOT/'configuration/workflow_thresholds.json')['thresholds'] if r['strategy']==a.strategy and r['development_target_percent']==a.budget)
 raw=kernels.candidate(d,kernels.uniform_ties(f),row);alert=kernels.cooldown(d,raw,a.days)
 assert not alert[d.hist].any();q=f[['episode_id','subject_id','t0']].copy();q['H90_flag']=d.hist;q['extra_candidate_before_suppression']=raw;q['extra_alert_after_suppression']=alert
 out.parent.mkdir(parents=True,exist_ok=True);q.to_csv(out,index=False,encoding='utf-8-sig');print('SIMULATED',len(q),'extra alerts',int(alert.sum()),'H90 alerts',int(d.hist.sum()))
if __name__=='__main__':main()
