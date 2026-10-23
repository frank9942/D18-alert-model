"""D18-specific completion; immutable parent models and source evidence."""
from pathlib import Path
import os,sys,json,time,gc,ast,copy,shutil,hashlib,importlib.util
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
S=Path(__file__).resolve().parent;R=S.parent;P=R.parent
S5=R/'05_完整对照与成人敏感性分析_20261014';S6=R/'06_首份申请检验消融病原体与标本_20261015';S7=R/'07_成人模型定向改进与重复提示_20261015';S8=R/'08_标本候选模型冻结与未知结局敏感性_20261015'
def module(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
parent=module('d18_parent7',S7/'common.py');base=parent.base
np,pd,duckdb,joblib=parent.np,parent.pd,parent.duckdb,parent.joblib
sha,read,now,parquet=parent.sha,parent.read,parent.now,parent.parquet
small,perf,lib=parent.small,parent.perf,parent.lib
CFG,FEATURES=parent.config('T0_specimen');COHORTS,BUDGETS,STRATEGIES,PATHS,EXTFEATURE=parent.COHORTS,parent.BUDGETS,parent.STRATEGIES,parent.PATHS,parent.EXTFEATURE
threadpool_limits=parent.threadpool_limits
LABS=['wbc_category','albumin_category','creatinine_category','crp_category','pct_category']
VARIANTS=['D18_no_labs','D18_availability_only','D18_after90']
CM=['new_positive_patients','positive_events','known_nonpositive_events','unknown_events','reviews_per_new_positive_patient','union_positive_patients','first_add','first_combined','first_coverage']
PM=['AUROC','AP','Brier','log_loss'];CAL=['calibration_in_the_large','calibration_joint_intercept','calibration_slope','observed_expected_ratio','Brier','Brier_skill_local_prevalence','mean_prediction','prevalence','AUROC','AP','log_loss']
ENDPOINTS=['full_24h','at_t0_all','at_t0_one'];CUTOFF='2024-02-29'
MAIN=S8/'models/D18_H3_Adult_T0spec_candidate_v1.joblib'
def save(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),'utf-8')
def csv(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def writeparquet(name,f):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True)
 with duckdb.connect() as c:c.from_df(f).write_parquet(str(p))
def functions(p,names,ns):
 tree=ast.parse(p.read_text('utf-8-sig'));body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in body}==set(names);exec(compile(ast.Module(body=body,type_ignores=[]),str(p),'exec'),ns);return ns
fast=functions(S7/'analyze.py',['context','choose_prefix','fast_value','calculate'],dict(np=np,small=small,CM=CM,BUDGETS=BUDGETS,STRATEGIES=STRATEGIES))
def protect():
 for p,h in read(S/'plan.json')['protected_hashes'].items():assert sha(p)==h,p
 return True
def load_frame(ci):
 f=parent.base.load_frame(ci);f=parent.attach_specimens(f,ci)
 sc=pd.read_csv(S7/f'private/scores_T0_specimen_{COHORTS[ci]}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip');assert sc.episode_id.is_unique
 sc=sc.set_index('episode_id').loc[f.episode_id.astype(str)]
 if 'subject_id' in sc:assert np.array_equal(sc.subject_id.astype(str),f.subject_id.astype(str))
 f['score']=sc.score.to_numpy(float);return f
def config(variant):
 c=copy.deepcopy(CFG);features=[x for x in FEATURES if x not in LABS] if variant=='D18_no_labs' else FEATURES.copy()
 if variant=='D18_availability_only':
  for lab in LABS:c['categorical_domains'][lab]=['AVAILABLE','MISSING']
 return c,features
def transform(f,variant):
 z=f.copy()
 if variant=='D18_availability_only':
  for lab in LABS:
   assert set(z[lab])<=set(CFG['categorical_domains'][lab]);z[lab]=np.where(z[lab].isin(['low','normal','high']),'AVAILABLE','MISSING')
 return z
def pipeline(variant,C):
 c,features=config(variant);return lib.pipeline(c,features,C)
def training_frame(variant):
 f=pd.read_csv(R/'数据准备/private/train_adult_matrix.csv',dtype=str,keep_default_na=False)
 for col in CFG['numeric_features']+['y']:f[col]=pd.to_numeric(f[col],errors='coerce')
 assert f.split.eq('train').all() and f.age.ge(18).all() and f.groupby('patient_id').fold.nunique().eq(1).all()
 f=parent.attach_specimens(f,0,'event_id')
 if variant=='D18_after90':f=f[pd.to_datetime(f.t0,utc=True)>=pd.Timestamp(CUTOFF,tz='Asia/Shanghai')].copy()
 return transform(f,variant).reset_index(drop=True)
def scored_variant(ci,variant):
 f=load_frame(ci);q=pd.read_csv(S/f'private/scores_{variant}_{COHORTS[ci]}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip');assert q.episode_id.is_unique
 if ci==0 and variant=='D18_after90':f=f[f.episode_id.astype(str).isin(q.episode_id)].copy()
 q=q.set_index('episode_id').loc[f.episode_id.astype(str)];assert len(q)==len(f)
 return f.assign(score=q.score.to_numpy(float)).reset_index(drop=True)
def freeze():
 assert not (S/'plan.json').exists()
 for n in ['models','private','results','audit','figures','checkpoints']:(S/n).mkdir(exist_ok=True)
 parent.protect()
 for stage in [S6,S7,S8]:
  manifest=stage/('audit/delivery_manifest.json' if stage==S6 else 'completion_manifest.json')
  for rel,h in read(manifest)['files'].items():assert sha(stage/rel)==(h['sha256'] if isinstance(h,dict) else h),(stage,rel)
 paths=PATHS+[EXTFEATURE,MAIN,R/'数据准备/private/train_adult_matrix.csv',R/'模型/D17_H3_Adult_v1.joblib',Path(base.__file__),Path(small.__file__),Path(perf.__file__),Path(lib.__file__),S7/'common.py',S7/'analyze.py',S7/'workflow.py',S7/'plan.json',S8/'plan.json']
 paths+=list(base.SPANPATHS.values())+list((S7/'private').glob('scores_T0_specimen_*.csv'))+list((S7/'private').glob('t0_specimens_*.parquet'))+list((S6/'private').glob('endpoints_*.parquet'))+list((S6/'private').glob('components_*.parquet'))
 paths+=list((S7/'models').glob('T0_specimen*.joblib'))+[S6/'audit/delivery_manifest.json',S7/'completion_manifest.json',S8/'completion_manifest.json']
 save('plan.json',dict(time=now(),status='FROZEN_BEFORE_D18_SPECIFIC_SUPPLEMENT_RESULTS',main='D18 frozen; D17 retained as historical reference; no refit of main',authorized_items=6,
 endpoint=dict(variants=ENDPOINTS,scores_fixed=True,no_retraining=True,first_ties='all T0 ties or same prespecified smallest hash single request from prior metadata',specimen_predictor='fixed co-T0 specimen category even in single-request outcome sensitivity; changing predictor would be separate model',history_and_candidate_pool_fixed=True),
 ablation=dict(variants=VARIANTS[:2],original_feature_count=18,no_labs_count=13,availability='five usable-by-T0 lab indicators plus remaining 13 predictors; not test-order flags'),
 training=dict(period='adult original development only',nested='same five patient folds, five outer/four inner; final fivefold',C_grid=CFG['learner']['C_grid'],criterion='mean log_loss, smaller C ties',fold_local_preprocessing=True,history_retrain_cutoff=CUTOFF,early_development_removed=True,legitimate_prior_history_retained=True),
 history=dict(spans='existing pre-T0 contact duration metadata; long >=90 days; short <90; unresolved if no usable span',continuous_observation_not_proven=True,ranking_and_quota_within_stratum=True),
 pathogen=dict(groups=['all_positive','H90_covered','D18_added','D18_only_vs_D17','D17_only_vs_D18'],evidence='OR target masks across selected positive events in each patient, not all organism records',exclusive_and_overlapping=True,zero_CRAB_does_not_validate_CRAB=True),
 workflow=dict(development_cutoffs_only=True,budgets=BUDGETS,strategies=['D18','icu_los'],days=[0,1,3,7],H90_unchanged=True,only_emitted_extra_alerts_reset=True,no_refill=True,no_outcomes_or_future_used=True,realized_workload_not_matched=True),
 calibration=dict(populations=['all_known','H90_unflagged_known'],CITL='offset-logit intercept slope fixed1',joint='intercept and slope freely estimated; report separately',no_recalibration=True,bootstrap_fitted_diagnostic_only=True,minimum_valid=950),
 bootstrap=dict(repeats=1000,seed=2026101401,workflow_seed=2026101501,paired_patient_clusters=True,fixed_model_scores=True,pointwise95=True,multiplicity_adjusted=False),
 interpretation=['exploratory; all later cohorts previously viewed','conditional CIs exclude model selection uncertainty','missing outcomes not negative','T0 restriction changes endpoint, not model development','new frozen supplements do not retrospectively make validation untouched'],
 protected_hashes={str(p):sha(p) for p in paths},material_passport=dict(origin_skill='academic-research-suite',origin_mode='experiment-agent/run',version_label='D18_specific_completion_20261015',verification_status='PLAN_FROZEN')))
 print('D18_SIX_ITEM_PLAN_FROZEN',flush=True)
if __name__=='__main__':freeze()

