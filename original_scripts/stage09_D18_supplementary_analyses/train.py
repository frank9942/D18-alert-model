from common import *
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
import warnings,sklearn

def train(variant):
 assert variant in VARIANTS and not (S/f'audit/fit_{variant}.json').exists();protect();start=time.monotonic()
 f=training_frame(variant);cfg,features=config(variant);y=f.y.to_numpy();known=np.isfinite(y)
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
 save(f'models/{variant}_definition.json',dict(features=features,configuration=parent.public_config(cfg,features),intercept=float(model.named_steps['model'].intercept_[0]),note='D18 derived model; T0 specimen retained; lab-availability transformation defined in common.py; after90 cutoff 2024-02-29. Serialized preprocessing.'))
 protect();save(f'audit/fit_{variant}.json',dict(status='PASS',selected=chosen,fits=fits,events=len(f),known_events=int(known.sum()),nested=True,evaluation_used=False,completed_utc=now(),seconds=time.monotonic()-start,hashes={str(p.relative_to(S)):sha(p) for p in list((S/'models').glob(variant+'*'))+[S/f'private/oof_{variant}.csv']}))

def score():
 protect()
 for variant in VARIANTS:
  if (S/f'audit/scores_{variant}.json').exists():continue
  fit=read(S/f'audit/fit_{variant}.json')
  for path,h in fit['hashes'].items():assert sha(S/path)==h
  model=joblib.load(S/f'models/{variant}.joblib');cfg,features=config(variant);paths=[]
  for ci,cname in enumerate(COHORTS):
   f=load_frame(ci)
   if ci==0:
    q=pd.read_csv(S/f'private/oof_{variant}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip')
    if variant=='D18_after90':f=f[f.episode_id.astype(str).isin(q.episode_id)].copy()
    q=q.set_index('episode_id').loc[f.episode_id.astype(str)];assert np.array_equal(q.subject_id.astype(str),f.subject_id.astype(str));scores=q.score.to_numpy()
   elif ci<3:scores=model.predict_proba(transform(f,variant)[features])[:,1]
   else:
    chunks=[]
    with duckdb.connect(config={'memory_limit':'256MB','threads':1}) as con:
     cur=con.execute('SELECT * FROM read_parquet(?)',[str(EXTFEATURE)])
     while True:
      x=cur.fetch_df_chunk(16)
      if x.empty:break
      for col in parent.CFG['categorical_domains']:x[col]=x[col].fillna('MISSING').astype(str).replace({'1.0':'1','0.0':'0','':'MISSING'})
      x=parent.attach_specimens(x,ci);assert x.age.ge(18).all()
      chunks.append(pd.DataFrame(dict(episode_id=x.episode_id.astype(str),score=model.predict_proba(transform(x,variant)[features])[:,1])))
    q=pd.concat(chunks,ignore_index=True);assert q.episode_id.is_unique and set(q.episode_id)==set(f.episode_id.astype(str));scores=q.set_index('episode_id').loc[f.episode_id.astype(str)].score.to_numpy()
   assert np.isfinite(scores).all();out=f[['episode_id','subject_id']].copy();out['score']=scores
   path=f'private/scores_{variant}_{cname}.csv';csv(path,out);paths.append(path);print('D18_SCORED',variant,cname,len(out),flush=True)
  save(f'audit/scores_{variant}.json',dict(status='PASS',time=now(),hashes={p:sha(S/p) for p in paths}))
 protect()
if __name__=='__main__':
 with threadpool_limits(limits=1):
  for v in VARIANTS:
   if not (S/f'audit/fit_{v}.json').exists():train(v)
  score()
