"""Portable D18 data contract and frozen-model access; no project paths or database connections."""
from pathlib import Path
import os,sys,json,hashlib
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
import kernels
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),'utf-8')
DEFINITION=read(ROOT/'configuration/current_D18_definition.json');CFG=DEFINITION['configuration'];FEATURES=DEFINITION['features'];REGISTRY=read(ROOT/'configuration/model_registry.json')
LABS=['wbc_category','albumin_category','creatinine_category','crp_category','pct_category']
SEX_ALIASES={'male':'男','female':'女','unknown':'未知'}  # English input values -> source labels inside the frozen models
REMOVED={'D18':[],'D18_no_labs':LABS,'D18_availability_only':[],'D18_after90':[],'D18_no_micro_history':['prior_target_reported_90d','prior_crgn_reported_90d'],'D18_no_ICU_LOS':['icu_status','length_of_stay']}
def config(variant):
 import copy
 c=copy.deepcopy(CFG);features=[f for f in FEATURES if f not in REMOVED[variant]]
 if variant=='D18_availability_only':
  for f in LABS:c['categorical_domains'][f]=['AVAILABLE','MISSING']
 return c,features
def prepare_features(raw,variant='D18'):
 missing=set(FEATURES)-set(raw.columns)
 if missing:raise ValueError('Missing D18 input columns: '+','.join(sorted(missing)))
 x=raw[FEATURES].copy()
 for name in CFG['numeric_features']:
  s=x[name].astype('string').str.strip();blank=s.isna()|s.isin(['','MISSING'])
  numeric=pd.to_numeric(s.mask(blank),errors='coerce')
  if (~blank&numeric.isna()).any():raise ValueError('Invalid numeric values in '+name)
  x[name]=numeric.astype(float)
  if np.isinf(x[name]).any():raise ValueError('Infinite values in '+name)
 if x.age.isna().any() or (x.age<18).any():raise ValueError('Adult inclusion requires known age >=18 at culture T0')
 if (x.length_of_stay.dropna()<0).any():raise ValueError('LOS must be elapsed days >=0, not total eventual stay')
 for name,domain in CFG['categorical_domains'].items():
  x[name]=x[name].astype('string').fillna('MISSING').str.strip().replace({'':'MISSING','1.0':'1','0.0':'0'})
  if name=='sex':x[name]=pd.Series([SEX_ALIASES.get(v.lower(),v) for v in x[name]],index=x.index,dtype='string')
  bad=set(x[name])-set(domain)
  if bad:raise ValueError('Unsupported category in '+name+': '+repr(sorted(bad))+'; follow schema; do not replace no-record by true absence')
 if variant=='D18_availability_only':
  for name in LABS:x[name]=np.where(x[name].isin(['low','normal','high']),'AVAILABLE','MISSING')
 return x

def load_model(variant='D18',fold=None):
 entry=REGISTRY[variant]
 if fold is not None:entry=next(x for x in entry['outer_models'] if x['fold']==fold)
 path=ROOT/entry['file']
 if sha(path)!=entry['sha256']:raise ValueError('Model checksum mismatch: '+entry['file'])
 return joblib.load(path)
def predict(raw,variant='D18',fold=None):
 model=load_model(variant,fold);x=prepare_features(raw,variant);features=list(model.feature_names_in_)
 with threadpool_limits(limits=1):p=model.predict_proba(x[features])[:,1]
 assert np.isfinite(p).all();return p

def load_events(path,cohort,labels_required=True):
 f=pd.read_csv(path,dtype=str,keep_default_na=False)
 req=['episode_id','subject_id','t0','score','length_of_stay','prior_crgn_reported_90d']
 if labels_required:req.append('y')
 for col in req:
  if col not in f:raise ValueError('Missing evaluation column: '+col)
 if not f.episode_id.is_unique or f.subject_id.eq('').any():raise ValueError('Unique episode IDs and nonempty patient IDs required')
 if not f.prior_crgn_reported_90d.isin(['1','MISSING']).all():raise ValueError('H90 coding must be 1 or MISSING')
 if 'y' not in f:f['y']=''
 if not f.y.isin(['','MISSING','0','1','0.0','1.0']).all():raise ValueError('y must be 0,1 or blank/MISSING; unknown is not 0')
 f['y']=pd.to_numeric(f.y.replace('MISSING',''),errors='coerce')
 f['score']=pd.to_numeric(f.score,errors='raise')
 if not (np.isfinite(f.score).all() and f.score.between(0,1,inclusive='neither').all()):raise ValueError('Scores must be finite and strictly between 0 and 1')
 los=f.length_of_stay.replace({'':'MISSING'});f['length_of_stay']=pd.to_numeric(los.replace('MISSING',np.nan),errors='raise')
 if f.length_of_stay.dropna().lt(0).any():raise ValueError('Negative LOS')
 col='icu_status' if 'icu_status' in f else 'icu'
 if col not in f:raise ValueError('icu_status or icu is required for fixed clinical comparator')
 if not f[col].isin(['0','1','MISSING','True','False','true','false']).all():raise ValueError('Invalid ICU status')
 f['icu']=f[col].isin(['1','True','true'])
 f['t0']=pd.to_datetime(f.t0,utc=True,errors='raise')
 if f.t0.isna().any():raise ValueError('T0 required')
 f['tie']=f.episode_id if cohort=='MIMIC_primary_adapted' else [hashlib.sha256(('crgn-rule-v1|'+s).encode()).hexdigest() for s in f.episode_id]
 return f.reset_index(drop=True)
COHORTS=['development_nested','hospital_2025','hospital_2026','MIMIC_primary_adapted']
