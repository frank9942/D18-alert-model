"""T0 specimen mapping, independent of labels and later requests."""
from common import *
from collections import defaultdict
def main():
 protect();assert not (S/'audit/preparation.json').exists();names=defaultdict(set)
 for line in DICT.read_text('utf-8-sig').splitlines():
  z=json.loads(line)
  if z['source_table']=='dbo.BT_Specimen':names[str(z['source_id'])].add(z['name_cn'])
 assert all(len(v)==1 for v in names.values());names={k:next(iter(v)) for k,v in names.items()}
 gd=pd.read_csv(GROUPING);gm={(r.source,r.raw_name):r.group for r in gd.itertuples()};gcode={'呼吸道':'RESP','血液':'BLOOD','尿液':'URINE','其他':'OTHER'}
 hu=parquet(Q/'private/hospital_unit_trace.parquet');hu.episode_id=hu.episode_id.astype(str);hu.unit_id=hu.unit_id.astype(str)
 earliest=hu[hu.time.eq(hu.t0)].copy();assert earliest.unit_id.is_unique;expected=dict(zip(earliest.unit_id,earliest.episode_id));rows=[]
 with STRICT.open(encoding='utf-8-sig') as fp:
  for line in fp:
   z=json.loads(line);rid=str(z['request_id'])
   if rid in expected:
    assert str(z['strict_event_id'])==expected[rid];rows.append(dict(episode_id=expected[rid],unit_id=rid,raw_name=names.get(str(z['specimen_id']),'UNMAPPED')))
 hm=pd.DataFrame(rows);assert hm.unit_id.is_unique and set(hm.unit_id)==set(expected)
 mu=parquet(Q/'private/MIMIC_unit_trace.parquet');mu.episode_id=mu.episode_id.astype(str);mu.unit_id=mu.unit_id.astype(str);me=mu[mu.time.eq(mu.t0)].copy();before=(DB.stat().st_size,DB.stat().st_mtime_ns)
 with duckdb.connect(str(DB),read_only=True,config={'memory_limit':'256MB','threads':1}) as c:
  c.register('earliest',me[['episode_id','unit_id']]);mm=c.execute('SELECT e.episode_id,e.unit_id,f.specimen_type raw_name FROM earliest e JOIN members m ON m.micro_specimen_id=e.unit_id AND CAST(m.episode_id AS VARCHAR)=e.episode_id JOIN frame_status f ON f.micro_specimen_id=e.unit_id').fetchdf()
 assert before==(DB.stat().st_size,DB.stat().st_mtime_ns) and mm.unit_id.is_unique and set(mm.unit_id)==set(me.unit_id)
 summary=[];distribution=[];trace=[]
 for ci,cname in enumerate(COHORTS):
  f=load_frame(ci);src=mm if ci==3 else hm;unit=me if ci==3 else earliest;allunit=mu if ci==3 else hu
  src=src[src.episode_id.isin(f.episode_id.astype(str))].copy();src['group']=[gcode.get(gm.get(('mimic' if ci==3 else 'hospital',str(n))),'MISSING') for n in src.raw_name]
  agg=src.groupby('episode_id').agg(groups=('group',lambda v:set(v)),raw_names=('raw_name',lambda v:' + '.join(sorted(set(str(x) for x in v)))),units=('unit_id','size')).loc[f.episode_id.astype(str)]
  assert len(agg)==len(f) and agg.units.ge(1).all();cats=['MISSING' if 'MISSING' in g else next(iter(g)) if len(g)==1 else 'MULTIPLE' for g in agg.groups]
  out=f[['episode_id','subject_id','t0']].copy();out['t0_specimen']=cats;out['co_earliest_requests']=agg.units.to_numpy();out['raw_names_at_t0']=agg.raw_names.to_numpy()
  chosen=unit[unit.episode_id.isin(f.episode_id.astype(str))];assert chosen.time.eq(chosen.t0).all();t=chosen.groupby('episode_id').t0.first().loc[f.episode_id.astype(str)]
  assert np.array_equal(pd.to_datetime(t,utc=True),pd.to_datetime(f.t0,utc=True));writeparquet(f'private/t0_specimens_{cname}.parquet',out)
  for cat,n in out.t0_specimen.value_counts().items():distribution.append(dict(cohort=cname,category=cat,events=int(n),fraction=n/len(f)))
  summary.append(dict(cohort=cname,events=len(f),t0_requests=len(src),co_earliest_multiple_requests=int(out.co_earliest_requests.gt(1).sum()),later_requests_excluded=int((allunit.episode_id.isin(f.episode_id.astype(str))&allunit.time.gt(allunit.t0)).sum()),missing_category=int(out.t0_specimen.eq('MISSING').sum()),patient_T0_match=True))
  trace.append(src.assign(cohort=cname));print('T0_SPECIMENS',summary[-1],flush=True)
 csv('results/t0_specimen_distribution.csv',distribution);csv('results/t0_specimen_mapping.csv',gd);writeparquet('private/t0_request_specimen_trace.parquet',pd.concat(trace,ignore_index=True))
 protect();save('audit/preparation.json',dict(status='PASS',time=now(),summary=summary,hospital_queries=0,cache_read_only_unchanged=True,derived_hashes={str(p.relative_to(S)):sha(p) for p in (S/'private').glob('*.parquet')}))
if __name__=='__main__':main()
