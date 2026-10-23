"""New candidate ranking under unknown labels; exhaustive bounds and paired bootstrap."""
from common import *
def selections(d,original,pct):
 ctx=d.context(np.ones(d.np,int));k=(pct*ctx['eligible_events']+99)//100
 ni,no=ctx['stratum_sizes'];ki=min(k,ni)
 exp=np.where(d.icu,ki/ni if ni else 0,(k-ki)/no if no else 0)*d.eligible
 return k,d.choose(ctx,'model',k)>0,dict(original_LR=original.choose(original.context(np.ones(original.np,int)),'model',k)>0,icu_los=d.choose(ctx,'icu_los',k)>0,icu_expected=exp)
def deterministic_bounds(d,m,c):
 def has(mask):return np.bincount(d.pi[mask],minlength=d.np)>0
 mp=has(m&d.pos);cp=has(c&d.pos);u=d.y<0;hu=has(d.hist&u)
 cats=[has(u&m&~c),has(u&~m&c),has(u&m&c)]
 lo=np.full(d.np,np.inf);hi=np.full(d.np,-np.inf)
 for bits in itertools.product([False,True],repeat=3):
  allowed=np.ones(d.np,bool)
  for j,b in enumerate(bits):
   if b:allowed&=cats[j]
  value=(mp|bits[0]|bits[2]).astype(int)-(cp|bits[1]|bits[2]).astype(int)
  lo=np.minimum(lo,np.where(allowed,value,np.inf));hi=np.maximum(hi,np.where(allowed,value,-np.inf))
 lo[hu]=np.minimum(lo[hu],0);hi[hu]=np.maximum(hi[hu],0);lo[d.base_pos]=0;hi[d.base_pos]=0
 delta=mp.astype(int)-cp.astype(int);delta[d.base_pos]=0
 assert np.all(lo<=delta)&np.all(hi>=delta)
 coef=m.astype(float)-c.astype(float);obs=coef[d.pos].sum()
 return dict(observed_patient_difference=float(delta.sum()),patient_difference_lower=float(lo.sum()),patient_difference_upper=float(hi.sum()),observed_positive_event_difference=float(obs),positive_event_difference_lower=float(obs+np.minimum(coef[u],0).sum()),positive_event_difference_upper=float(obs+np.maximum(coef[u],0).sum()),bounds_type='sharp_identification_bounds_not_CI')
def first_bounds(d,m,sel):
 first=base.first_mask(d.f,d.pos);order=d.f.sort_values(['t0','episode_id'],kind='stable').index.to_numpy();rank=np.empty(len(d.f),int);rank[order]=np.arange(len(d.f))
 fr=np.full(d.np,len(d.f),int);fr[d.pi[first]]=rank[first];feasible=first|((d.y<0)&(rank<fr[d.pi]))
 gain=m.astype(float)-sel.astype(float);lo=np.full(d.np,np.inf);hi=np.full(d.np,-np.inf)
 np.minimum.at(lo,d.pi[feasible],gain[feasible]);np.maximum.at(hi,d.pi[feasible],gain[feasible]);none=~d.all_positive_patients
 lo[none]=np.minimum(lo[none],0);hi[none]=np.maximum(hi[none],0)
 observed=float(gain[first].sum());assert lo.sum()<=observed+1e-8<=hi.sum()+1e-8
 return dict(observed_first_difference=observed,first_difference_lower=float(lo.sum()),first_difference_upper=float(hi.sum()),patients_with_earlier_or_only_unknown=int(np.unique(d.pi[feasible&~first]).size),bounds_type='sharp_identification_bounds_not_CI')
def tests():
 qa=sharpns['verify']();rng=np.random.default_rng(2026101517);assignments=0
 for iteration in range(60):
  ne=int(rng.integers(3,9));f=pd.DataFrame(dict(episode_id=np.arange(ne),subject_id=rng.integers(0,3,ne),t0=pd.date_range('2025-01-01',periods=ne),y=rng.choice([0,1,np.nan],ne),icu=rng.random(ne)<.6,score=rng.random(ne),tie=np.arange(ne),prior_crgn_reported_90d=np.where(rng.random(ne)<.25,'1','MISSING'),length_of_stay=rng.integers(0,20,ne)))
  d=small.Cohort(f);original=small.Cohort(f.assign(score=rng.random(ne)));u=np.flatnonzero(d.y<0);pct=50;k,m,ss=selections(d,original,pct)
  cb={n:deterministic_bounds(d,m,c) for n,c in ss.items() if n!='icu_expected'};fb={n:first_bounds(d,m,c) for n,c in ss.items()};vals={n:[] for n in ss};fv={n:[] for n in ss}
  for bits in itertools.product([0,1],repeat=len(u)):
   z=f.copy();z.loc[u,'y']=bits;dd=small.Cohort(z);ctx=dd.context(np.ones(dd.np,int));fm=base.first_mask(z,dd.pos)
   for n,sel in ss.items():
    if n=='icu_expected':v=dd.evaluate(ctx,'icu_expected',k)['new_positive_patients']
    else:
     p=np.bincount(dd.pi[sel&dd.pos],minlength=dd.np)>0;v=float((p&~dd.base_pos).sum())
    pm=np.bincount(dd.pi[m&dd.pos],minlength=dd.np)>0;mv=float((pm&~dd.base_pos).sum())
    vals[n].append(mv-v);fv[n].append(float((m.astype(float)-sel.astype(float))[fm].sum()))
   assignments+=1
  for n,r in cb.items():assert np.isclose(min(vals[n]),r['patient_difference_lower']) and np.isclose(max(vals[n]),r['patient_difference_upper'])
  for n,r in fb.items():assert np.isclose(min(fv[n]),r['first_difference_lower']) and np.isclose(max(fv[n]),r['first_difference_upper'])
 return dict(status='PASS',inherited_expected_bounds_and_allocation=qa,additional_cases=60,additional_exhaustive_assignments=assignments,comparators=['original_LR','icu_los','icu_expected'],first_positive_recomputed=True)
def calc(ds,pw,rep=None):
 obs=fast['calculate'](ds['observed'],pw);positive=fast['calculate'](ds['all_unknown_positive'],pw)
 negative=[]
 for row in obs:
  z=row.copy();z['known_nonpositive_events']=z['known_nonpositive_events']+z['unknown_events'];z['unknown_events']=0.;negative.append(z)
 return [dict(scenario=scenario,replicate=rep,**r) for scenario,rr in [('observed',obs),('all_unknown_nonpositive',negative),('all_unknown_positive',positive)] for r in rr]
def profile(ci,f,d,original,q):
 out=[]
 for pct in BUDGETS:
  k,m,ss=selections(d,original,pct);ss['T0_specimen']=m
  for name,sel in [('all_events',np.ones(len(f))),('H90',d.hist),('H90_unflagged',d.eligible)]+list(ss.items()):
   for cat in parent.SPEC_DOMAIN:
    mask=q.t0_specimen.to_numpy()==cat;n=float(sel[mask].sum())
    if n==0:continue
    pos=float(sel[mask&d.pos].sum());neg=float(sel[mask&d.neg].sum());uk=float(sel[mask&(d.y<0)].sum())
    out.append(dict(cohort=COHORTS[ci],budget_percent=pct,population=name,t0_specimen=cat,events=n,positive_events=pos,known_nonpositive_events=neg,unknown_events=uk,unknown_fraction=uk/n))
 return out
def audit_specimens():
 trace=parquet(S7/'private/t0_request_specimen_trace.parquet');trace.episode_id=trace.episode_id.astype(str);trace.unit_id=trace.unit_id.astype(str)
 checks=[];mapping=pd.read_csv(parent.GROUPING);gm={(r.source,r.raw_name):r.group for r in mapping.itertuples()};gcode={'呼吸道':'RESP','血液':'BLOOD','尿液':'URINE','其他':'OTHER'}
 for domain,which in [('hospital',[0,1,2]),('MIMIC',[3])]:
  units=parquet(parent.Q/f'private/{domain}_unit_trace.parquet');units.episode_id=units.episode_id.astype(str);units.unit_id=units.unit_id.astype(str)
  for ci in which:
   f,ff,q=frames(ci);t=trace[trace.cohort.eq(COHORTS[ci])].copy();assert t.unit_id.is_unique
   u=units[units.episode_id.isin(f.episode_id.astype(str))]
   expected=u[u.time.eq(u.t0)]
   assert set(t.unit_id)==set(expected.unit_id)
   j=t.merge(expected[['unit_id','episode_id','time','t0']],on=['unit_id','episode_id'],validate='one_to_one')
   assert len(j)==len(t) and j.time.eq(j.t0).all()
   mapped=[gcode.get(gm.get(('mimic' if ci==3 else 'hospital',str(n))),'MISSING') for n in j.raw_name]
   assert np.array_equal(j.group,np.array(mapped))
   ag=j.groupby('episode_id').group.agg(set).loc[f.episode_id.astype(str)]
   cats=['MISSING' if 'MISSING' in g else next(iter(g)) if len(g)==1 else 'MULTIPLE' for g in ag]
   assert np.array_equal(cats,q.t0_specimen)
   counts=j.groupby('episode_id').size().loc[f.episode_id.astype(str)].to_numpy()
   assert np.array_equal(counts,q.co_earliest_requests.to_numpy())
   checks.append(dict(cohort=COHORTS[ci],events=len(f),requests_at_T0=len(t),later_requests_excluded=int(u.time.gt(u.t0).sum()),category_recomputed_mismatches=0,subject_and_T0_match=True,unmapped_events=int((np.array(cats)=='MISSING').sum())))
 save('audit/specimen_recheck.json',dict(status='PASS',time=now(),checks=checks,label_used_in_mapping=False,new_database_queries=0,scope='Independent reconstruction from frozen S7 trace, prior request timestamps and mapping dictionary; raw source extraction inherited and hash protected',limits=['No historical field revision log','MIMIC charttime proxy for order time']))
 return checks
def run():
 protect();start=time.monotonic()
 if not (S/'audit/bounds_math.json').exists():save('audit/bounds_math.json',tests())
 if not (S/'audit/specimen_recheck.json').exists():audit_specimens()
 print('MATH_AND_TIMING_VERIFIED',flush=True)
 allpt=[];alliv=[];cb=[];fb=[];profiles=[];baseline=[]
 for ci,name in enumerate(COHORTS):
  checkpoint=S/f'checkpoints/{name}.json'
  if checkpoint.exists():
   chk=read(checkpoint)
   for rel,h in chk['hashes'].items():assert sha(S/rel)==h
   for fname,dst in [('points',allpt),('intervals',alliv),('cumulative_bounds',cb),('first_bounds',fb),('specimen_unknown_profile',profiles),('baselines',baseline)]:dst.extend(pd.read_csv(S/f'results/{fname}_{name}.csv').to_dict('records'))
   continue
  f,ft,q=frames(ci);ds={}
  for scenario in ['observed','all_unknown_positive']:
   ds[scenario]={}
   for model,z in [('original_LR',f),('T0_specimen',ft)]:
    if scenario=='all_unknown_positive':z=z.assign(y=z.y.fillna(1))
    d=small.Cohort(z);d.first=base.first_mask(d.f,d.pos);ds[scenario][model]=d
  d=ds['observed']['T0_specimen'];original=ds['observed']['original_LR'];ones=np.ones(d.np,int)
  pts=calc(ds,ones);bnds=[];first=[];prof=profile(ci,f,d,original,q);bs=[]
  for scenario,models in ds.items():
   z=models['T0_specimen'];bs.append(dict(cohort=name,scenario=scenario,events=len(f),patients=d.np,positive_events=int(z.pos.sum()),positive_patients=int(z.all_positive_patients.sum()),unknown_events=int((z.y<0).sum()),H90_covered_positive_patients=int(z.base_pos.sum()),first_H90=int((z.first&z.hist).sum())))
  for pct in BUDGETS:
   k,m,ss=selections(d,original,pct)
   for comp,sel in ss.items():
    bb=sharpns['sharp'](d.pi,d.y,d.hist,m,d.icu,k,d.np) if comp=='icu_expected' else deterministic_bounds(d,m,sel)
    bnds.append(dict(cohort=name,budget_percent=pct,comparator=comp,**bb));first.append(dict(cohort=name,budget_percent=pct,comparator=comp,**first_bounds(d,m,sel)))
  for r in pts:r['cohort']=name
  old=pd.read_csv(S7/f'results/bounded_clinical_points_{name}.csv')
  ncheck=0
  for r in pts:
   if r['scenario']!='observed':continue
   ref=old[(old.variant==r['variant'])&(old.budget_percent==r['budget_percent'])&(old.strategy==r['strategy'])].iloc[0]
   for key in CM:
    if key in r and pd.notna(ref.get(key)):
     assert np.isclose(r[key],ref[key],rtol=0,atol=1e-8),(name,r['variant'],key);ncheck+=1
  csv(f'results/points_{name}.csv',pts)
  for tag,data in [('cumulative_bounds',bnds),('first_bounds',first),('specimen_unknown_profile',prof),('baselines',bs)]:csv(f'results/{tag}_{name}.csv',data)
  print('POINTS',name,'2percent bounds',[(x['comparator'],x['patient_difference_lower'],x['patient_difference_upper']) for x in bnds if x['budget_percent']==2],flush=True)
  rng=np.random.default_rng(2026101401+ci);boots=[]
  for rep in range(1,1001):
   pw=np.bincount(rng.integers(0,d.np,d.np),minlength=d.np)
   boots.extend(dict(cohort=name,**r) for r in calc(ds,pw,rep))
   if rep%100==0:print('BOOT',name,rep,'elapsed',round(time.monotonic()-start,1),flush=True)
  bt=pd.DataFrame(boots);pt=pd.DataFrame(pts)
  observed=bt[bt.scenario.eq('observed')].set_index(['replicate','variant','budget_percent','strategy']).sort_index()
  prior=pd.read_csv(S7/f'private/bootstrap_bounded_clinical_{name}.csv').set_index(['replicate','variant','budget_percent','strategy']).sort_index().loc[observed.index]
  bootchecks=0
  for key in CM:
   a=observed[key].to_numpy(float);b=prior[key].to_numpy(float);assert np.allclose(a,b,rtol=0,atol=1e-8,equal_nan=True),(name,key);bootchecks+=len(a)
  neg=bt[bt.scenario.eq('all_unknown_nonpositive')].set_index(['replicate','variant','budget_percent','strategy']).sort_index()
  assert np.allclose(observed.new_positive_patients,neg.new_positive_patients) and np.allclose(observed.first_add,neg.first_add)
  assert np.allclose(neg.known_nonpositive_events,observed.known_nonpositive_events+observed.unknown_events)
  iv=base.summarize(pt,bt,['cohort','scenario','variant','budget_percent','strategy'],CM)
  csv(f'results/intervals_{name}.csv',iv);csv(f'private/bootstrap_{name}.csv',bt)
  names=[f'results/{tag}_{name}.csv' for tag in ['points','intervals','cumulative_bounds','first_bounds','specimen_unknown_profile','baselines']]+[f'private/bootstrap_{name}.csv']
  save(f'checkpoints/{name}.json',dict(status='PASS',time=now(),replicates=1000,observed_point_values_reproduced=ncheck,observed_bootstrap_values_reproduced=bootchecks,all_negative_algebra_verified=True,hashes={p:sha(S/p) for p in names}))
  for dst,data in [(allpt,pts),(alliv,iv),(cb,bnds),(fb,first),(profiles,prof),(baseline,bs)]:dst.extend(data)
  del ds,d,original,boots,bt,pt,prior,observed,neg;gc.collect()
 for name,data in [('scenario_points',allpt),('scenario_intervals',alliv),('cumulative_sharp_bounds',cb),('first_sharp_bounds',fb),('specimen_unknown_profile',profiles),('scenario_baselines',baseline)]:csv(f'results/{name}.csv',data)
 protect();save('audit/analysis_complete.json',dict(status='PASS',time=now(),elapsed_seconds=time.monotonic()-start,cohorts=COHORTS,budgets=BUDGETS,replicates=1000,source_observed_bootstrap_reproduced=True,models_and_parent_labels_unmodified=True,no_probability_imputation=True))
 print('ALL_SENSITIVITIES_COMPLETE',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if len(sys.argv)>1 and sys.argv[1]=='test':print(tests(),flush=True)
  else:run()

