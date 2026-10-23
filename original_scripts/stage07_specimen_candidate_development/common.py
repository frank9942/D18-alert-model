"""Adult bounded modifications. Frozen parent inputs are read-only."""
from pathlib import Path
import os,sys,json,copy,time,gc,importlib.util
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
S=Path(__file__).resolve().parent;R=S.parent;P=R.parent
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=module('adult_s5_fixed',R/'05_完整对照与成人敏感性分析_20261014/common.py')
np,pd,duckdb,joblib=base.np,base.pd,base.duckdb,base.joblib
sha,read,now,parquet,load_frame=base.sha,base.read,base.now,base.parquet,base.load_frame
small,perf,lib,CFG,FEATURES=base.small,base.perf,base.lib,base.CFG,base.FEATURES
COHORTS,BUDGETS,STRATEGIES,PATHS,EXTFEATURE=base.COHORTS,base.BUDGETS,base.STRATEGIES,base.PATHS,base.EXTFEATURE
threadpool_limits=base.threadpool_limits
VARIANTS=['LOS_log1p','H90_unflagged','T0_specimen']
Q=P/'outputs/v7_first_request_sensitivity_20261009_v1'
STRICT=P/'database/postgresql/specialty_compact/严格队列差异与培养用途核对_20260925_v2/strict_members.jsonl'
DICT=P/'database/postgresql/specialty_compact/python_etl/runs/smoke_3_core/baseline/tables/dict_item.jsonl'
GROUPING=P/'outputs/v7_publication_tables_20261009_v1/tables/11_标本展示分组字典.csv'
DB=P/'outputs/mimic_external_audit_20261005/private/audit.duckdb'
SPEC_DOMAIN=['RESP','BLOOD','URINE','OTHER','MULTIPLE','MISSING']
def save(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),'utf-8')
def csv(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def writeparquet(name,f):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True)
 with duckdb.connect() as c:c.from_df(f).write_parquet(str(p))
def protect():
 for p,h in read(S/'plan.json')['protected_hashes'].items():assert sha(p)==h,p
 return True
def config(variant):
 c=copy.deepcopy(CFG);features=FEATURES.copy()
 if variant=='H90_unflagged':features.remove('prior_crgn_reported_90d')
 if variant=='T0_specimen':
  features.append('t0_specimen');c['features'].append('t0_specimen');c['categorical_domains']['t0_specimen']=SPEC_DOMAIN
 return c,features
def pipeline(variant,C):
 from sklearn.compose import ColumnTransformer
 from sklearn.pipeline import Pipeline
 from sklearn.preprocessing import FunctionTransformer,StandardScaler,OneHotEncoder
 from sklearn.impute import SimpleImputer
 cfg,features=config(variant);m=lib.pipeline(cfg,features,C)
 if variant=='LOS_log1p':
  cats=[f for f in features if f not in cfg['numeric_features']]
  m.steps[0]=('preprocess',ColumnTransformer([
   ('age',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),['age']),
   ('los',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('log1p',FunctionTransformer(np.log1p,feature_names_out='one-to-one')),('scale',StandardScaler())]),['length_of_stay']),
   ('category',OneHotEncoder(categories=[cfg['categorical_domains'][f] for f in cats],sparse_output=False,handle_unknown='error'),cats)],remainder='drop'))
 return m
def attach_specimens(f,ci,id_col='episode_id'):
 q=parquet(S/f'private/t0_specimens_{COHORTS[ci]}.parquet');assert q.episode_id.is_unique
 q.episode_id=q.episode_id.astype(str);q=q.set_index('episode_id').loc[f[id_col].astype(str)];z=f.copy();z['t0_specimen']=q.t0_specimen.to_numpy();return z
def freeze():
 assert not (S/'plan.json').exists()
 for n in ['models','private','results','audit','figures','checkpoints']:(S/n).mkdir(exist_ok=True)
 paths=PATHS+[EXTFEATURE,R/'数据准备/private/train_adult_matrix.csv',R/'模型/D17_H3_Adult_v1.joblib',R/'方案与确认/成人研究方案_冻结.json',R/'历史来源/feature_domain_plan.json',R/'历史来源/parent_model_library_reference_only.py',Path(base.__file__),Path(small.__file__),Path(perf.__file__),STRICT,DICT,GROUPING]
 paths+=list((R/'结果').glob('*.csv'))+[Q/'private/hospital_unit_trace.parquet',Q/'private/MIMIC_unit_trace.parquet']
 save('plan.json',dict(created_utc=now(),status='FROZEN_BEFORE_NEW_FITS',primary_retained=True,scope='adult frozen membership/labels/H90; no live hospital queries',
 variants={'LOS_log1p':'replace LOS by log(1+LOS); age remains linear, no spline search','H90_unflagged':'fit known-outcome development EVENTS without H90 at T0; remove constant prior_crgn_reported_90d; never exclude patients using future history','T0_specimen':'one fixed-domain field from co-earliest requests only; no later 24h specimens'},
 specimen=dict(domain=SPEC_DOMAIN,groups='existing clinical dictionary; >=2 broad groups -> MULTIPLE; unmapped -> MISSING',hospital='first culture order; exact ties retained',mimic='specimen charttime proxy, not order timestamp; historical specimen fields lack revision history'),
 tuning=dict(C_grid=CFG['learner']['C_grid'],metric='minimum mean log loss, smaller C ties',nested='same adult patient folds 5 outer / 4 inner; final fivefold',preprocessing='fit-fold only',winner_selection='none from temporal/external results; separate changes, no combination'),
 evaluation=dict(budgets=BUDGETS,comparators=['original_LR','icu_expected','icu_los'],primary='new positive patients beyond whole-period H90 coverage',complement='first known-positive event capture regardless of later H90',unknown='occupies capacity, never imputed as negative',performance='AUROC AP Brier log-loss calibration; H90-only model performance restricted to H90-unflagged known events'),
 bootstrap=dict(repeats=1000,seed=2026101401,unit='paired patient cluster',models_fixed=True,rerank_quota=True,interval='pointwise95percentile',multiplicity_adjusted=False),
 suppression=dict(separate=True,strategies=['original_LR','icu_los'],days=[0,1,3,7],thresholds='development nested OOF score and development ICU/LOS cutoffs for 1/2/5 percent of H90-unflagged events; different realized workload in later cohorts',sequence='chronological per patient; only emitted extra alerts start cooldown; suppressed alerts do not extend it; exact boundary eligible',history='H90 flags remain unchanged and reported separately; suppress extra alerts only, H90 does not reset cooldown',refill=False,outcome_used=False,interpretation='workflow simulation, no equal-workload superiority claim or preferred clinical interval; no MIMIC calendar daily rates'),
 interpretation=['exploratory after prior evaluations seen','no equivalence, deployment-readiness or clinical acceptability claim'],
 material_passport=dict(origin_skill='academic-research-suite',origin_mode='experiment-agent/run',version_label='Adult_v1_bounded_extensions',verification_status='PLAN_FROZEN'),
 protected_hashes={str(p):sha(p) for p in paths},initial_code_hashes={p.name:sha(p) for p in S.glob('*.py')}))
 print('BOUNDED_PLAN_FROZEN',flush=True)
if __name__=='__main__':freeze()

def public_config(cfg,features):
 return dict(features=features,numeric_features=[f for f in cfg['numeric_features'] if f in features],categorical_domains={k:v for k,v in cfg['categorical_domains'].items() if k in features},learner=cfg['learner'],never_predictors=cfg['never_predictors'],current_cohort='Frozen adult cohort; actual counts and fitted population in audit/before_fit_<variant>.json',label_source='Frozen adult matrix outcome y, inherited v7; no outcome changes in this round',configuration_source='Historical feature-domain/learner template only; old template cohort counts and label-version fields intentionally omitted')
