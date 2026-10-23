"""Nested adult bounded modifications. Evaluation cohorts never enter fit/tuning."""
from common import *
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
import warnings,sklearn
def train(variant):
 assert variant in VARIANTS and not (S/f'audit/fit_{variant}.json').exists();protect();start=time.monotonic()
 for p,h in read(S/'audit/preparation.json')['derived_hashes'].items():assert sha(S/p)==h
 f=pd.read_csv(R/'数据准备/private/train_adult_matrix.csv',dtype=str,keep_default_na=False)
 for col in CFG['numeric_features']+['y']:f[col]=pd.to_numeric(f[col],errors='coerce')
 assert f.split.eq('train').all() and f.age.ge(18).all() and f.groupby('patient_id').fold.nunique().eq(1).all()
 f=attach_specimens(f,0,'event_id') if variant=='T0_specimen' else f;cfg,features=config(variant);y=f.y.to_numpy();known=np.isfinite(y)
 if variant=='H90_unflagged':known&=~f.prior_crgn_reported_90d.eq('1').to_numpy()
 fd=f.fold.astype(float).astype(int).to_numpy();oof=np.full(len(f),np.nan)
 save(f'audit/before_fit_{variant}.json',dict(time=now(),plan_sha256=sha(S/'plan.json'),code_sha256=sha(Path(__file__)),input_sha256=sha(R/'数据准备/private/train_adult_matrix.csv'),python=sys.version,sklearn=sklearn.__version__,variant=variant,features=features,events=len(f),known_events=int(known.sum()),evaluation_used=False))
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
   print('BOUNDED_FIT',variant,outer,'C',best,'fits',fits,'seconds',round(time.monotonic()-start,1),flush=True)
 assert np.isfinite(oof).all();csv(f'private/oof_{variant}.csv',dict(episode_id=f.event_id,subject_id=f.patient_id,score=oof))
 csv(f'results/coefficients_{variant}.csv',dict(encoded_feature=model.named_steps['preprocess'].get_feature_names_out(),coefficient=model.named_steps['model'].coef_[0]))
 save(f'models/{variant}_definition.json',dict(features=features,configuration=public_config(cfg,features),intercept=float(model.named_steps['model'].intercept_[0]),note='All imputers, scales and category encoders are serialized in joblib; LOS transformation serialized in joblib; T0 category defined in prepare.py; H90_unflagged intended only for unflagged events'))
 protect();save(f'audit/fit_{variant}.json',dict(status='PASS',selected=chosen,fits=fits,events=len(f),known_events=int(known.sum()),nested=True,evaluation_used=False,completed_utc=now(),seconds=time.monotonic()-start,hashes={str(p.relative_to(S)):sha(p) for p in list((S/'models').glob(variant+'*'))+[S/f'private/oof_{variant}.csv']}))
def score():
 protect();missing=[]
 for variant in VARIANTS:
  if (S/f'audit/scores_{variant}.json').exists():
   for p,h in read(S/f'audit/scores_{variant}.json')['hashes'].items():assert sha(S/p)==h
   continue
  fit=read(S/f'audit/fit_{variant}.json');assert fit['status']=='PASS'
  for p,h in fit['hashes'].items():assert sha(S/p)==h
  m=joblib.load(S/f'models/{variant}.joblib');cfg,features=config(variant);paths=[]
  for ci,cname in enumerate(COHORTS):
   if ci<3:
    f=load_frame(ci)
    if variant=='T0_specimen':f=attach_specimens(f,ci)
    if ci==0:
     q=pd.read_csv(S/f'private/oof_{variant}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip').set_index('episode_id').loc[f.episode_id];assert np.array_equal(q.subject_id,f.subject_id);values=q.score.to_numpy()
    else:values=m.predict_proba(f[features])[:,1]
    out=pd.DataFrame(dict(episode_id=f.episode_id,subject_id=f.subject_id,score=values))
   else:
    chunks=[]
    with duckdb.connect(config={'memory_limit':'128MB','threads':1}) as con:
     cur=con.execute('SELECT * FROM read_parquet(?)',[str(EXTFEATURE)])
     while True:
      x=cur.fetch_df_chunk(8)
      if x.empty:break
      for col in CFG['categorical_domains']:x[col]=x[col].fillna('MISSING').astype(str).replace({'1.0':'1','0.0':'0','':'MISSING'})
      assert x.age.ge(18).all()
      x.episode_id=x.episode_id.astype(str)
      if variant=='T0_specimen':x=attach_specimens(x,ci)
      chunks.append(pd.DataFrame(dict(episode_id=x.episode_id,score=m.predict_proba(x[features])[:,1])))
    out=pd.concat(chunks,ignore_index=True)
   assert out.episode_id.is_unique and np.isfinite(out.score).all();path=f'private/scores_{variant}_{cname}.csv';csv(path,out);paths.append(path);print('BOUNDED_SCORED',variant,cname,flush=True)
  save(f'audit/scores_{variant}.json',dict(status='PASS',time=now(),hashes={p:sha(S/p) for p in paths}))
 protect()
if __name__=='__main__':
 with threadpool_limits(limits=1):
  if sys.argv[1]=='score':score()
  else:train(sys.argv[1])
