"""Pathogen coverage from D18-selected positive events; explicit overlapping groups."""
from common import *
def run():
 protect();exclusive=[];overlap=[];coverage=[];membership=[];exchange=[]
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);d=small.Cohort(f);ctx=d.context(np.ones(d.np,int))
  cm=parquet(S6/f'private/components_{cname}.parquet');cm.episode_id=cm.episode_id.astype(str);cm=cm.set_index('episode_id').loc[f.episode_id.astype(str)]
  assert np.array_equal(cm.subject_id.astype(str),f.subject_id.astype(str)) and np.allclose(cm.y,f.y,equal_nan=True)
  mask=cm.component_mask.to_numpy(int);assert np.array_equal(mask>0,d.pos)
  original=small.Cohort(base.load_frame(ci));oc=original.context(np.ones(original.np,int));assert np.array_equal(original.pi,d.pi) and np.array_equal(original.hist,d.hist)
  hp=set(d.patients[d.base_pos]);groups=[('all_positive',0,d.pos),('H90_covered',0,d.pos&d.hist)];summ={}
  for pct in BUDGETS:
   k=(pct*ctx['eligible_events']+99)//100;a=d.choose(ctx,'model',k)>0;b=original.choose(oc,'model',k)>0
   an=a&d.pos&~d.base_pos[d.pi];bn=b&d.pos&~d.base_pos[d.pi];ap=set(d.pi[an]);bp=set(d.pi[bn])
   groups += [('D18_added',pct,an),('D18_only_vs_D17',pct,an&np.isin(d.pi,list(ap-bp))),('D17_only_vs_D18',pct,bn&np.isin(d.pi,list(bp-ap)))]
   exchange.append(dict(cohort=cname,budget_percent=pct,D18_added=len(ap),D17_added=len(bp),both=len(ap&bp),D18_only=len(ap-bp),D17_only=len(bp-ap),net=len(ap)-len(bp)))
   assert len(ap)==d.evaluate(ctx,'model',k)['new_positive_patients']
  for group,pct,sel in groups:
   pm=np.zeros(d.np,int);np.bitwise_or.at(pm,d.pi[sel],mask[sel]);summ[group,pct]=pm;ids=np.flatnonzero(pm>0);total=len(ids)
   cats={'CRE_only':pm==1,'CRPA_only':pm==2,'CRAB_only':pm==4,'multiple_target_groups':np.isin(pm,[3,5,6,7])}
   assert sum(int(v.sum()) for v in cats.values())==total
   for cat,z in cats.items():exclusive.append(dict(cohort=cname,group=group,budget_percent=pct,category=cat,patients=int(z.sum()),total_patients=total,fraction=float(z.sum()/total) if total else np.nan))
   for cat,bit in [('CRE',1),('CRPA',2),('CRAB',4)]:overlap.append(dict(cohort=cname,group=group,budget_percent=pct,pathogen=cat,patients=int(((pm&bit)>0).sum()),total_patients=total,positive_events=int((sel&((mask&bit)>0)).sum()),selected_positive_events=int(sel.sum())))
   membership.extend(dict(cohort=cname,group=group,budget_percent=pct,subject_id=d.patients[j],component_mask=int(pm[j])) for j in ids)
  for pct in BUDGETS:
   for cat,bit in [('CRE',1),('CRPA',2),('CRAB',4)]:
    total=int(((summ['all_positive',0]&bit)>0).sum());h=int(((summ['H90_covered',0]&bit)>0).sum());added=int(((summ['D18_added',pct]&bit)>0).sum())
    assert h+added<=total
    coverage.append(dict(cohort=cname,budget_percent=pct,pathogen=cat,all_positive_patients=total,H90_covered_patients=h,D18_added_patients=added,combined=h+added,H90_fraction=h/total if total else np.nan,combined_fraction=(h+added)/total if total else np.nan,added_fraction_of_all=added/total if total else np.nan,zero_denominator=total==0))
 for n,data in [('pathogen_patient_exclusive',exclusive),('pathogen_patient_overlapping',overlap),('pathogen_group_coverage',coverage),('D18_D17_patient_exchange',exchange)]:csv(f'results/{n}.csv',data)
 csv('private/pathogen_patient_membership.csv',membership);protect()
 save('audit/pathogens.json',dict(status='PASS',time=now(),source_components_reused=True,D18_selection_recomputed=True,three_budgets=True,all_positive_masks_reconciled=True,groups_overlap=True,exclusive_groups_sum_to_total=True,evidence='only selected positive events; OR per patient, multi-group may span events',no_new_infection_or_treatment_claim=True))
 print('D18_PATHOGENS_COMPLETE',flush=True)
if __name__=='__main__':run()

