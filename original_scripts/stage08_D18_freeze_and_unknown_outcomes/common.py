"""Frozen T0 specimen candidate: new unknown-outcome sensitivity, no new fits."""
from pathlib import Path
import os,sys,json,ast,time,gc,itertools,shutil,importlib.util
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
S=Path(__file__).resolve().parent;R=S.parent;P=R.parent
S7=R/'07_成人模型定向改进与重复提示_20261015'
def module(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
parent=module('frozen_s7',S7/'common.py');base=parent.base
np,pd,duckdb,joblib=parent.np,parent.pd,parent.duckdb,parent.joblib
sha,read,now,parquet,load_frame=parent.sha,parent.read,parent.now,parent.parquet,parent.load_frame
small=parent.small;COHORTS=parent.COHORTS;BUDGETS=parent.BUDGETS;STRATEGIES=parent.STRATEGIES
threadpool_limits=parent.threadpool_limits
CM=['new_positive_patients','positive_events','known_nonpositive_events','unknown_events','reviews_per_new_positive_patient','union_positive_patients','first_add','first_combined','first_coverage']
SCENARIOS=['observed','all_unknown_nonpositive','all_unknown_positive']
CANDIDATE='D18_H3_Adult_T0spec_candidate_v1'
def save(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),'utf-8')
def csv(name,x):
 p=S/name;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def functions(p,names,ns):
 tree=ast.parse(p.read_text('utf-8-sig'));body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert {n.name for n in body}==set(names)
 exec(compile(ast.Module(body=body,type_ignores=[]),str(p),'exec'),ns);return ns
fast=functions(S7/'analyze.py',['context','choose_prefix','fast_value','calculate'],dict(np=np,small=small,CM=CM,BUDGETS=BUDGETS,STRATEGIES=STRATEGIES))
sharpns=functions(P/'outputs/v7_unknown_components_20261007_v1/unknown.py',['cnt','sharp','literal','verify'],dict(np=np,small=small,itertools=itertools))
def protect():
 for p,h in read(S/'plan.json')['protected_hashes'].items():assert sha(p)==h,p
 assert sha(S/f'models/{CANDIDATE}.joblib')==sha(S7/'models/T0_specimen.joblib')
def frames(ci):
 f=load_frame(ci)[['episode_id','subject_id','t0','y','icu','tie','score','length_of_stay','prior_crgn_reported_90d']]
 sc=pd.read_csv(S7/f'private/scores_T0_specimen_{COHORTS[ci]}.csv',dtype={'episode_id':str,'subject_id':str},float_precision='round_trip')
 assert sc.episode_id.is_unique;sc=sc.set_index('episode_id').loc[f.episode_id.astype(str)]
 if 'subject_id' in sc:assert np.array_equal(sc.subject_id.astype(str),f.subject_id.astype(str))
 q=parquet(S7/f'private/t0_specimens_{COHORTS[ci]}.parquet');q.episode_id=q.episode_id.astype(str);q=q.set_index('episode_id').loc[f.episode_id.astype(str)]
 assert np.array_equal(q.subject_id.astype(str),f.subject_id.astype(str))
 assert np.array_equal(pd.to_datetime(q.t0,utc=True),pd.to_datetime(f.t0,utc=True))
 return f,f.assign(score=sc.score.to_numpy(float)),q
def freeze():
 assert not (S/'plan.json').exists()
 for name in ['models','private','results','audit','figures','checkpoints']:(S/name).mkdir(exist_ok=True)
 prior=read(S7/'completion_manifest.json')
 for rel,h in prior['files'].items():assert sha(S7/rel)==h,rel
 paths=list(parent.PATHS)+[R/'模型/D17_H3_Adult_v1.joblib',S7/'plan.json',S7/'completion_manifest.json',S7/'models/T0_specimen.joblib',S7/'models/T0_specimen_definition.json',S7/'common.py',S7/'analyze.py',S7/'prepare.py',Path(base.__file__),Path(small.__file__),P/'outputs/v7_unknown_components_20261007_v1/unknown.py',parent.Q/'private/hospital_unit_trace.parquet',parent.Q/'private/MIMIC_unit_trace.parquet']
 paths+=list((S7/'private').glob('scores_T0_specimen_*.csv'))+list((S7/'private').glob('t0_*.parquet'))+list((S7/'models').glob('T0_specimen_outer*.joblib'))
 paths+=list((S7/'results').glob('bounded_clinical*.csv'))+list((S7/'private').glob('bootstrap_bounded_clinical_*.csv'))
 paths += [S7/'audit'/n for n in ['preparation.json','verification.json','fit_T0_specimen.json','cv_T0_specimen.json','scores_T0_specimen.json','before_fit_T0_specimen.json']]
 shutil.copy2(S7/'models/T0_specimen.joblib',S/f'models/{CANDIDATE}.joblib')
 definition=read(S7/'models/T0_specimen_definition.json')
 save(f'models/{CANDIDATE}_definition.json',dict(candidate_name=CANDIDATE,status='FROZEN_EXPLORATORY_CANDIDATE_NOT_PRIMARY_OR_DEPLOYED',model_sha256=sha(S/f'models/{CANDIDATE}.joblib'),source=str(S7/'models/T0_specimen.joblib'),selection_disclosure='Candidate designation follows viewed S7 performance; no claim of prospective or untouched validation.',features=definition['features'],configuration=definition['configuration'],intercept=definition['intercept'],specimen_domain=parent.SPEC_DOMAIN,specimen_rule='All co-earliest requests at T0 only; >=2 broad types MULTIPLE; any unmapped MISSING; charttime proxy in MIMIC',model_definition_source=str(S7/'models/T0_specimen_definition.json'),C=0.1,score_provenance='Development uses nested patient out-of-fold scores from five outer models; later cohorts use the single final frozen model. This step does not refit or recalibrate.'))
 shutil.copy2(S7/'results/coefficients_T0_specimen.csv',S/'models/coefficients.csv')
 for n in ['preparation.json','verification.json','fit_T0_specimen.json','cv_T0_specimen.json','scores_T0_specimen.json','before_fit_T0_specimen.json']:shutil.copy2(S7/'audit'/n,S/'audit'/('source_'+n))
 save('plan.json',dict(time=now(),status='FROZEN_BEFORE_THIS_SENSITIVITY_NOT_BEFORE_CANDIDATE_SELECTION',candidate=CANDIDATE,primary='D17_H3_Adult_v1 retained',no_new_fitting=True,no_live_queries=True,cohorts=COHORTS,budgets=BUDGETS,comparators=['original_LR','icu_expected','icu_los'],scenarios=SCENARIOS,unknown=dict(scores_and_H90_flags_fixed=True,labels_changed_only_in_memory=True,first_positive_recomputed=True,all_negative_positive_not_joint_extrema=True,sharp_bounds='original-sample sharp identification bounds; not confidence intervals; per-metric endpoints need not share same assignment',probabilistic_imputation=False),bootstrap=dict(repeats=1000,seed=2026101401,unit='paired patients',fixed_scores=True,rerank_and_quota=True,CI='pointwise95percentile',multiplicity_adjusted=False,does_not_include='model selection, development or post hoc design uncertainty'),limitations=['All evaluation cohorts previously viewed; exploratory','No clinical acceptability or deployment threshold','MIMIC charttime proxy and unknown outcomes','Candidate final model is not the development OOF score generator'],protected_hashes={str(p):sha(p) for p in paths},material_passport=dict(origin_skill='academic-research-suite',origin_mode='experiment-agent',version_label='Adult_T0_candidate_unknown_sensitivity',verification_status='FROZEN_PLAN')))
 protect();print('CANDIDATE_AND_SENSITIVITY_PLAN_FROZEN',CANDIDATE,flush=True)
if __name__=='__main__':freeze()

