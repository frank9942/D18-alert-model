"""Attach an already frozen T0-specimen mapping without round-tripping numeric tokens."""
from runtime import *
import argparse,duckdb

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--mapping',required=True);p.add_argument('--id-column',choices=['event_id','episode_id'],default='event_id');p.add_argument('--out',required=True);a=p.parse_args();dest=Path(a.out)
 if dest.exists():raise FileExistsError('Use a new output file')
 raw=pd.read_csv(a.input,dtype=str,keep_default_na=False)
 if 't0_specimen' in raw:raise ValueError('Input already has t0_specimen; do not overwrite silently')
 if a.id_column not in raw or not raw[a.id_column].is_unique:raise ValueError('Unique input event IDs required')
 mp=Path(a.mapping)
 if mp.suffix.lower()=='.parquet':
  with duckdb.connect() as c:q=c.execute('SELECT episode_id,t0_specimen FROM read_parquet(?)',[str(mp)]).fetchdf()
 else:q=pd.read_csv(mp,dtype=str,keep_default_na=False)[['episode_id','t0_specimen']]
 q.episode_id=q.episode_id.astype(str)
 if not q.episode_id.is_unique:raise ValueError('Duplicate mapping keys')
 q=q.set_index('episode_id');ids=raw[a.id_column].astype(str)
 if not ids.isin(q.index).all():raise ValueError('Mapping must cover every input event; missing keys are not a clinical MISSING category')
 raw['t0_specimen']=q.loc[ids,'t0_specimen'].to_numpy();prepare_features(raw)
 dest.parent.mkdir(parents=True,exist_ok=True);raw.to_csv(dest,index=False,encoding='utf-8-sig')
 save(dest.with_suffix('.provenance.json'),dict(input_sha256=sha(a.input),mapping_sha256=sha(mp),output_sha256=sha(dest),rows=len(raw),numeric_input_tokens_preserved=True,new_clinical_mapping_inferred=False))
 print('ATTACHED',len(raw),'frozen T0 categories; original numeric tokens preserved')
if __name__=='__main__':main()
