"""Two prespecified D18 group ablations; first-layer H90 stays unchanged."""
from pathlib import Path
import os,sys,json,time,gc,copy,shutil,importlib.util
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
S=Path(__file__).resolve().parent;R=S.parent;P=R.parent
S9=R/'09_D18专属补充分析_20261015'
def module(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
original=module('d18_six_supplement',S9/'common.py')
for n in ['parent','base','np','pd','duckdb','joblib','sha','read','now','parquet','small','perf','lib','CFG','FEATURES','COHORTS','BUDGETS','STRATEGIES','PATHS','EXTFEATURE','threadpool_limits','MAIN','CM','PM','fast','S7','S8']:
 globals()[n]=getattr(original,n)
VARIANTS=['D18_no_micro_history','D18_no_ICU_LOS']
REMOVED={'D18_no_micro_history':['prior_target_reported_90d','prior_crgn_reported_90d'],'D18_no_ICU_LOS':['icu_status','length_of_stay']}
def save(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),'utf-8')
def csv(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def protect():
 for p,h in read(S/'plan.json')['protected_hashes'].items():assert sha(p)==h,p
 return True
load_frame=original.load_frame
transform=lambda f,variant:f.copy()
def config(variant):
 assert variant in VARIANTS;c=copy.deepcopy(CFG);features=[f for f in FEATURES if f not in REMOVED[variant]]
 assert len(features)==16 and 't0_specimen' in features
 return c,features
def pipeline(variant,C):
 cfg,features=config(variant);return lib.pipeline(cfg,features,C)
def training_frame(variant):
 assert variant in VARIANTS;return original.training_frame('D18')
def scored_variant(ci,variant):
 f=load_frame(ci);q=pd.read_csv(S/f'private/scores_{variant}_{COHORTS[ci]}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip');assert q.episode_id.is_unique and len(q)==len(f)
 q=q.set_index('episode_id').loc[f.episode_id.astype(str)];assert np.array_equal(q.subject_id.astype(str),f.subject_id.astype(str))
 return f.assign(score=q.score.to_numpy(float)).reset_index(drop=True)
def freeze():
 assert not (S/'plan.json').exists()
 original.protect();parent.protect()
 manifest=read(S9/'completion_manifest.json');assert manifest['status']=='COMPLETE_VERIFIED'
 for rel,h in manifest['files'].items():assert sha(S9/rel)==h,rel
 for d in ['models','private','results','audit','figures','checkpoints']:(S/d).mkdir(exist_ok=True)
 paths=list(read(S9/'plan.json')['protected_hashes'])+[S9/'completion_manifest.json',S9/'common.py',S9/'train.py',S9/'analyze.py',S9/'verify.py',S9/'results/ablation_clinical_points.csv',S9/'results/ablation_clinical_intervals.csv',S9/'results/ablation_performance_points.csv',S9/'results/ablation_performance_intervals.csv']
 save('plan.json',dict(status='FROZEN_BEFORE_GROUP_ABLATION_FITS_AND_RESULTS',time=now(),variants=REMOVED,main_D18_unchanged=True,first_layer_H90_unchanged=True,H90_not_zeroed=True,cohorts_and_labels_and_quotas_unchanged=True,training=dict(same_all_development_known_events=True,patient_folds='5 outer / 4 inner; final fivefold',C_grid=CFG['learner']['C_grid'],selection='mean log_loss; smaller C ties',preprocessing='training-fold only',model_family='L2 logistic',features_per_ablation=16),evaluation=dict(cohorts=COHORTS,budgets=BUDGETS,performance_populations=['all_known','H90_unflagged_known'],performance=['AUROC','AP','Brier','log_loss','calibration point diagnostics'],clinical=['additional positive patients beyond H90','first known-positive capture','known nonpositive alerts','unknown alerts','positive event alerts','reviews per additional patient'],comparators=['frozen D18','ICU priority','ICU + LOS'],bootstrap=1000,paired_patient_clusters=True,scores_fixed=True,rerank_each_sample=True,seed=2026101401,pointwise95=True,multiplicity_adjustment=False),interpretation=['exploratory, previously viewed later cohorts','ablation estimates predictive dependence, not causal clinical importance','H90 is constant unflagged in second-layer pool, but training on all development can change coefficients','no removal of first-layer history rule','no winner selection using later cohorts','intervals exclude training and model-selection uncertainty','no equivalence claims from CI crossing zero'],protected_hashes={str(p):sha(p) for p in paths},initial_code_hashes={p.name:sha(p) for p in S.glob('*.py')},material_passport=dict(origin_skill='academic-research-suite',origin_mode='experiment-agent/run',version_label='D18_group_ablation_20261015',verification_status='PLAN_FROZEN')))
 print('D18_GROUP_ABLATION_PLAN_FROZEN',flush=True)
if __name__=='__main__':freeze()
