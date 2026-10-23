"""Reproduce development-only nested fitting from a prepared feature matrix with fixed patient folds."""
from runtime import *
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime,timezone
import time,warnings,sklearn,argparse
VARIANTS=list(REMOVED)
def now():return datetime.now(timezone.utc).isoformat()
def save(name,x):
 import runtime
 runtime.save(S/name,x)
def csv(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def protect():
 assert sha(INPUT)==INPUT_HASH,'Training input changed during run'
def public_config(cfg,features):return dict(features=features,numeric_features=[f for f in features if f in cfg['numeric_features']],categorical_domains={f:cfg['categorical_domains'][f] for f in features if f not in cfg['numeric_features']},learner=cfg['learner'],never_predictors=cfg['never_predictors'])
def training_frame(variant):
 raw=pd.read_csv(INPUT,dtype=str,keep_default_na=False)
 for c in ['event_id','patient_id','t0','split','fold','y']:
  if c not in raw:raise ValueError('Missing training field: '+c)
 if not raw.split.eq('train').all():raise ValueError('Only development split=train may enter fitting')
 if not raw.event_id.is_unique or raw.patient_id.eq('').any():raise ValueError('Invalid event/patient IDs')
 if not raw.y.isin(['','MISSING','0','1','0.0','1.0']).all():raise ValueError('Invalid development outcome')
 x=prepare_features(raw,variant)
 for c in FEATURES:raw[c]=x[c]
 raw['y']=pd.to_numeric(raw.y.replace('MISSING',''),errors='coerce');raw['fold']=pd.to_numeric(raw.fold,errors='raise')
 if not raw.fold.isin(range(5)).all() or not raw.groupby('patient_id').fold.nunique().eq(1).all():raise ValueError('Patient-isolated integer folds 0..4 are required')
 times=pd.to_datetime(raw.t0,utc=True,errors='raise')
 start=pd.Timestamp('2023-12-01',tz='Asia/Shanghai');end=pd.Timestamp('2025-01-01',tz='Asia/Shanghai')
 if not ((times>=start)&(times<end)).all():raise ValueError('Rows outside original adult development period')
 if variant=='D18_after90':raw=raw[times>=pd.Timestamp('2024-02-29',tz='Asia/Shanghai')].copy()
 save('training_input_audit.json',dict(source_sha256=sha(INPUT),variant=variant,retained_events=len(raw),known_events=int(raw.y.notna().sum()),patient_folds_reused=True,features=config(variant)[1],evaluation_data_used=False))
 return raw.reset_index(drop=True)
def pipeline(variant,C):
 cfg,features=config(variant);return kernels.pipeline(cfg,features,C)
def train(variant):
 assert variant in VARIANTS and not (S/f'audit/fit_{variant}.json').exists();protect();start=time.monotonic()
 f=training_frame(variant);cfg,features=config(variant);y=f.y.to_numpy();known=np.isfinite(y)
 fd=f.fold.astype(float).astype(int).to_numpy();oof=np.full(len(f),np.nan)
 save(f'audit/before_fit_{variant}.json',dict(time=now(),plan_sha256=sha(S/'plan.json'),code_sha256=sha(Path(__file__)),input_sha256=sha(INPUT),python=sys.version,sklearn=sklearn.__version__,variant=variant,features=features,events=len(f),known_events=int(known.sum()),evaluation_used=False))
 warnings.filterwarnings('error',category=ConvergenceWarning);cv=[];chosen=[];fits=0
 with threadpool_limits(limits=1):
  for outer in [0,1,2,3,4,None]:
   eligible=known&(fd!=outer) if outer is not None else known.copy();folds=[j for j in range(5) if j!=outer];cache=[]
   for inner in folds:
    a=eligible&(fd!=inner);b=eligible&(fd==inner);assert not set(f.loc[a,'patient_id'])&set(f.loc[b,'patient_id'])
    pp=pipeline(variant,.1).named_steps['preprocess'];cache.append((pp.fit_transform(f.loc[a,features]),y[a].astype(int),pp.transform(f.loc[b,features]),y[b].astype(int)))
   candidates=[]
   for c in cfg['learner']['C_grid']:
    losses=[]
    for xa,ya,xb,yb in cache:
     m=pipeline(variant,c).named_steps['model'];m.fit(xa,ya);losses.append(float(log_loss(yb,m.predict_proba(xb)[:,1],labels=[0,1])));fits+=1
    row=dict(outer='final' if outer is None else outer,C=c,fold_losses=losses,mean_log_loss=float(np.mean(losses)));cv.append(row);candidates.append(row)
   best=min(candidates,key=lambda r:(round(r['mean_log_loss']/1e-12)*1e-12,r['C']))['C'];model=pipeline(variant,best);model.fit(f.loc[eligible,features],y[eligible].astype(int));fits+=1
   path=S/f"models/{variant}{'' if outer is None else '_outer'+str(outer)}.joblib";joblib.dump(model,path)
   if outer is not None:
    test=fd==outer;assert not set(f.loc[eligible,'patient_id'])&set(f.loc[test,'patient_id']);oof[test]=model.predict_proba(f.loc[test,features])[:,1]
    assert np.allclose(oof[test],joblib.load(path).predict_proba(f.loc[test,features])[:,1],rtol=0,atol=1e-12)
   chosen.append(dict(outer='final' if outer is None else outer,C=best,sha256=sha(path)));save(f'audit/cv_{variant}.json',cv)
   print('PORTABLE_FIT',variant,outer,'C',best,'fits',fits,'seconds',round(time.monotonic()-start,1),flush=True)
 assert np.isfinite(oof).all();csv(f'private/oof_{variant}.csv',dict(episode_id=f.event_id,subject_id=f.patient_id,score=oof))
 csv(f'results/coefficients_{variant}.csv',dict(encoded_feature=model.named_steps['preprocess'].get_feature_names_out(),coefficient=model.named_steps['model'].coef_[0]))
 save(f'models/{variant}_definition.json',dict(features=features,configuration=public_config(cfg,features),intercept=float(model.named_steps['model'].intercept_[0]),removed=REMOVED[variant],note='Portable nested refit of the chosen frozen feature specification. Model family and development-only selection unchanged; see training_input_audit.json for actual rows.'))
 protect();save(f'audit/fit_{variant}.json',dict(status='PASS',selected=chosen,fits=fits,events=len(f),known_events=int(known.sum()),nested=True,evaluation_used=False,completed_utc=now(),seconds=time.monotonic()-start,hashes={str(p.relative_to(S)):sha(p) for p in list((S/'models').glob(variant+'*'))+[S/f'private/oof_{variant}.csv']}))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--variant',choices=VARIANTS,default='D18');a=p.parse_args()
 INPUT=Path(a.input);INPUT_HASH=sha(INPUT);S=Path(a.out)
 if S.exists() and any(S.iterdir()):raise FileExistsError('Use an empty output directory; frozen package models must not be overwritten')
 for name in ['audit','models','private','results']:(S/name).mkdir(parents=True,exist_ok=True)
 save('plan.json',dict(status='PORTABLE_DEVELOPMENT_REFIT',variant=a.variant,source_sha256=INPUT_HASH,main_model_hash=REGISTRY['D18']['sha256'],schema_hash=sha(ROOT/'configuration/current_D18_definition.json'),code_sha256=sha(Path(__file__)),evaluation_used=False))
 with threadpool_limits(limits=1):train(a.variant)
