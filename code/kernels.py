"""Exact audited mathematical definitions; paths/data readers intentionally excluded."""
import numpy as np,pandas as pd,hashlib,itertools
from types import SimpleNamespace
from scipy.stats import hypergeom
from scipy.special import expit,logit
from scipy.optimize import minimize,brentq
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
BUDGETS=[1,2,5]
STRATEGIES=['model','icu_expected','icu_los']
CM=['new_positive_patients','positive_events','known_nonpositive_events','unknown_events','reviews_per_new_positive_patient','union_positive_patients','first_add','first_combined','first_coverage']
PM=['AUROC','AP','Brier','log_loss']
def ratio(a, b):
    return float(a / b) if b else None

def no_hit(n, k, m):
    """Exact without-replacement zero-hit probabilities using stable products."""
    m = np.asarray(m, dtype=int)
    assert 0 <= k <= n and np.all(m >= 0)
    if k == 0:
        return np.ones(m.shape, dtype=float)
    limit = min(int(m.max(initial=0)), n - k)
    table = np.zeros(int(m.max(initial=0)) + 1, dtype=float)
    table[0] = 1.0
    if limit:
        table[1:limit + 1] = np.exp(np.cumsum(np.log1p(-k / (n - np.arange(limit)))))
    return table[m]

class Cohort:
    def __init__(self, f):
        self.f = f.reset_index(drop=True).copy()
        f = self.f
        assert f.episode_id.is_unique and f.subject_id.notna().all()
        self.y = f.y.fillna(-1).to_numpy(dtype=int)
        assert set(self.y) <= {-1, 0, 1}
        self.pos, self.neg = self.y == 1, self.y == 0
        self.hist = f.prior_crgn_reported_90d.eq('1').to_numpy()
        self.eligible = ~self.hist
        self.icu = f.icu.to_numpy(dtype=bool)
        self.score = f.score.to_numpy(dtype=float)
        assert np.isfinite(self.score).all()
        self.los = pd.to_numeric(f.length_of_stay, errors='coerce').to_numpy(dtype=float)
        assert not np.any(self.los[np.isfinite(self.los)] < 0)
        self.pi, self.patients = pd.factorize(f.subject_id, sort=True)
        self.np = len(self.patients)
        self.base_pos = np.bincount(self.pi[self.hist & self.pos], minlength=self.np) > 0
        self.base_any = np.bincount(self.pi[self.hist], minlength=self.np) > 0
        self.all_positive_patients = np.bincount(self.pi[self.pos], minlength=self.np) > 0
        tie = f.tie.to_numpy()
        los_key = np.where(np.isfinite(self.los), self.los, -np.inf)
        orders = {
            'model': np.lexsort((tie, -self.score)),
            'icu_los': np.lexsort((tie, -los_key, -self.icu.astype(int))),
        }
        self.orders = {s: order[self.eligible[order]] for s, order in orders.items()}
        self.strata = [self.eligible & self.icu, self.eligible & ~self.icu]
        self.patient_counts = []
        for mask in self.strata:
            self.patient_counts.append({
                'all': np.bincount(self.pi[mask], minlength=self.np),
                'pos': np.bincount(self.pi[mask & self.pos], minlength=self.np),
                'neg': np.bincount(self.pi[mask & self.neg], minlength=self.np),
                'unknown': np.bincount(self.pi[mask & (self.y < 0)], minlength=self.np),
            })

    def context(self, pw):
        w = pw[self.pi]
        return {
            'pw': pw, 'w': w,
            'eligible_events': int(w[self.eligible].sum()),
            'eligible_positive_events': int(w[self.eligible & self.pos].sum()),
            'eligible_known_nonpositive_events': int(w[self.eligible & self.neg].sum()),
            'uncovered_positive_patients_total': int(pw[self.all_positive_patients & ~self.base_pos].sum()),
            'baseline_positive_patients': int(pw[self.base_pos].sum()),
            'baseline_alerts': int(w[self.hist].sum()),
            'stratum_sizes': [int(w[mask].sum()) for mask in self.strata],
        }

    def choose(self, ctx, strategy, k):
        order = self.orders[strategy]
        weights = ctx['w'][order]
        cumulative = np.cumsum(weights)
        take = np.minimum(weights, np.maximum(0, k - (cumulative - weights)))
        out = np.zeros(len(self.f), dtype=int)
        out[order] = take
        assert int(out.sum()) == k
        return out

    def finish(self, ctx, k, tp, fp, unknown, new_patients, reached, newly_reached):
        assert np.isclose(tp + fp + unknown, k, atol=1e-8)
        assert new_patients <= ctx['uncovered_positive_patients_total'] + 1e-8
        return {
            'added_alerts': k, 'positive_events': float(tp),
            'known_nonpositive_events': float(fp), 'unknown_events': float(unknown),
            'new_positive_patients': float(new_patients),
            'baseline_positive_patients': ctx['baseline_positive_patients'],
            'union_positive_patients': ctx['baseline_positive_patients'] + float(new_patients),
            'ppv_known': ratio(tp, tp + fp), 'false_discovery_known': ratio(fp, tp + fp),
            'false_positive_rate_known': ratio(fp, ctx['eligible_known_nonpositive_events']),
            'unknown_fraction': ratio(unknown, k), 'confirmed_yield_all_alerts': ratio(tp, k),
            'reviews_per_new_positive_patient': ratio(k, new_patients),
            'new_patient_coverage_fraction': ratio(new_patients, ctx['uncovered_positive_patients_total']),
            'conditional_positive_event_recall': ratio(tp, ctx['eligible_positive_events']),
            'supplement_distinct_patients': float(reached),
            'supplement_new_patients_beyond_any_h90_alert': float(newly_reached),
            'extra_events_per_reached_patient': ratio(k, reached),
            'eligible_events': ctx['eligible_events'],
            'uncovered_positive_patients_total': ctx['uncovered_positive_patients_total'],
        }

    def deterministic(self, ctx, strategy, k):
        sw = self.choose(ctx, strategy, k)
        assert not sw[self.hist].any() and np.all(sw <= ctx['w'])
        positive_capture = np.zeros(self.np, dtype=int)
        np.maximum.at(positive_capture, self.pi[self.pos], sw[self.pos])
        capture = np.zeros(self.np, dtype=int)
        selected = sw > 0
        np.maximum.at(capture, self.pi[selected], sw[selected])
        return self.finish(ctx, k, sw[self.pos].sum(), sw[self.neg].sum(), sw[self.y < 0].sum(),
                           positive_capture[~self.base_pos].sum(), capture.sum(), capture[~self.base_any].sum())

    def expected(self, ctx, strategy, k):
        if strategy == 'random_expected':
            groups = [{key: sum(x[key] for x in self.patient_counts) for key in self.patient_counts[0]}]
            sizes, ks = [ctx['eligible_events']], [k]
        else:
            groups = self.patient_counts
            sizes = ctx['stratum_sizes']
            ks = [min(k, sizes[0]), max(0, k - sizes[0])]
        no_positive, no_any = np.ones(self.np), np.ones(self.np)
        tp, fp, unknown = 0.0, 0.0, 0.0
        for g, n, take in zip(groups, sizes, ks):
            assert 0 <= take <= n
            if take:
                tp += take * np.dot(ctx['pw'], g['pos']) / n
                fp += take * np.dot(ctx['pw'], g['neg']) / n
                unknown += take * np.dot(ctx['pw'], g['unknown']) / n
            no_positive *= no_hit(n, take, g['pos'])
            no_any *= no_hit(n, take, g['all'])
        new = np.dot(ctx['pw'][~self.base_pos], (1 - no_positive)[~self.base_pos])
        reached = np.dot(ctx['pw'], 1 - no_any)
        newly_reached = np.dot(ctx['pw'][~self.base_any], (1 - no_any)[~self.base_any])
        return self.finish(ctx, k, tp, fp, unknown, new, reached, newly_reached)

    def evaluate(self, ctx, strategy, k):
        return self.deterministic(ctx, strategy, k) if strategy in self.orders else self.expected(ctx, strategy, k)

def literal_counts(d, copies, selected):
    base = {(d.pi[j], copy) for j, copy in copies if d.hist[j] and d.pos[j]}
    base_any = {(d.pi[j], copy) for j, copy in copies if d.hist[j]}
    positive = {(d.pi[j], copy) for j, copy in selected if d.pos[j]}
    reached = {(d.pi[j], copy) for j, copy in selected}
    return np.array([sum(d.pos[j] for j, copy in selected),
                     sum(d.neg[j] for j, copy in selected),
                     sum(d.y[j] < 0 for j, copy in selected),
                     len(positive - base), len(reached), len(reached - base_any)], dtype=float)

def verify_math():
    f = pd.DataFrame({
        'episode_id': np.arange(8), 'subject_id': ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'd'],
        'y': [1, 1, 1, 1, 0, 0, np.nan, np.nan],
        'prior_crgn_reported_90d': ['1'] + ['MISSING'] * 7,
        'icu': [True, True, True, False, True, False, True, False],
        'score': [.9, .8, .7, .7, .7, .4, .3, .2], 'tie': np.arange(8),
        'length_of_stay': [4., 5., 1., 6., np.nan, 7., 3., 1.],
    })
    d = Cohort(f)
    keys = ['positive_events', 'known_nonpositive_events', 'unknown_events', 'new_positive_patients',
            'supplement_distinct_patients', 'supplement_new_patients_beyond_any_h90_alert']
    exact_tests = deterministic_tests = 0
    for pw in [np.ones(d.np, dtype=int), np.array([2, 1, 2, 1])]:
        ctx = d.context(pw)
        copies = [(j, copy) for j in range(len(f)) for copy in range(int(pw[d.pi[j]]))]
        eligible = [x for x in copies if d.eligible[x[0]]]
        for k in sorted({0, 1, 3, len(eligible) - 1, len(eligible)}):
            for strategy in ['random_expected', 'icu_expected']:
                fixed = []
                if strategy == 'random_expected':
                    pool, take = eligible, k
                else:
                    icu = [x for x in eligible if d.icu[x[0]]]
                    if k <= len(icu):
                        pool, take = icu, k
                    else:
                        fixed = icu
                        pool, take = [x for x in eligible if not d.icu[x[0]]], k - len(icu)
                literal = np.mean([literal_counts(d, copies, fixed + list(draw))
                                   for draw in itertools.combinations(pool, take)], axis=0)
                result = d.expected(ctx, strategy, k)
                assert np.allclose(literal, [result[key] for key in keys], atol=1e-10), (strategy, k)
                exact_tests += 1
            for strategy, order in d.orders.items():
                ranked = [(j, copy) for j in order for copy in range(int(pw[d.pi[j]]))]
                literal = literal_counts(d, copies, ranked[:k])
                result = d.deterministic(ctx, strategy, k)
                assert np.array_equal(literal, [result[key] for key in keys])
                deterministic_tests += 1
    for n, k in [(1, 0), (1, 1), (10, 3), (310977, 3088), (77903, 15438)]:
        m = np.arange(min(n, 100) + 1)
        assert np.allclose(no_hit(n, k, m), hypergeom.pmf(0, n, m, k), atol=1e-10)
    return {'status': 'PASS', 'exhaustive_random_expectation_checks': exact_tests,
            'explicit_patient_copy_deterministic_checks': deterministic_tests,
            'scipy_hypergeometric_reference_cases': 5,
            'covered_cases': ['unknown-only patient', 'baseline-covered patient', 'repeated positive events',
                              'patient resampled twice', 'mixed ICU strata', 'K=0', 'K=N', 'missing LOS', 'score ties']}
allocation_math_test=verify_math
def calibration(scores,total,pos):
    """Grouped binomial MLE, algebraically equivalent to weighted event fits."""
    z=logit(np.clip(scores,1e-6,1-1e-6)); n=float(total.sum())
    if not (0 < pos.sum() < n):raise ArithmeticError('no two outcome classes')
    a=0.0
    for it in range(100):
        q=expit(z+a);den=np.sum(total*q*(1-q))
        if den<1e-12:raise ArithmeticError('degenerate offset calibration')
        step=float(np.clip(np.sum(pos-total*q)/den,-5,5));a+=step
        if abs(step)<1e-9:break
    else:raise ArithmeticError('offset calibration nonconvergence')
    if np.ptp(z)<1e-10:raise ArithmeticError('constant scores: slope undefined')
    center=np.average(z,weights=total);x=z-center
    beta=np.array([center,1.0])
    def loss(b):
        eta=b[0]+b[1]*x
        return np.sum(total*np.logaddexp(0,eta)-pos*eta)
    for it in range(100):
        q=expit(beta[0]+beta[1]*x);v=total*q*(1-q);r=pos-total*q
        h=np.array([[v.sum(),(v*x).sum()],[(v*x).sum(),(v*x*x).sum()]])
        step=np.linalg.solve(h,np.array([r.sum(),(r*x).sum()]))
        scale=1.0;before=loss(beta)
        while scale>2**-20 and loss(beta+scale*step)>before+1e-9:scale/=2
        beta+=scale*step
        if np.max(np.abs(scale*step))<1e-9:break
    else:raise ArithmeticError('joint calibration nonconvergence')
    q=expit(beta[0]+beta[1]*x)
    assert max(abs(np.sum(pos-total*q))/n,abs(np.sum((pos-total*q)*x))/n)<1e-7
    return {'calibration_in_the_large':float(a),'calibration_joint_intercept':float(beta[0]-beta[1]*center),
            'calibration_slope':float(beta[1])}

class Evaluator:
    def __init__(self,f):
        self.f=f.reset_index(drop=True)
        self.y=f.y.to_numpy(float);self.p=f.score.to_numpy(float);self.pid=f['_pid'].to_numpy(int)
        assert set(np.unique(self.y))=={0.,1.} and np.isfinite(self.p).all() and ((self.p>0)&(self.p<1)).all()
        self.scores,self.g=np.unique(self.p,return_inverse=True)
        self.ng=len(self.scores)
        self.edges=np.unique(np.quantile(self.p,np.linspace(0,1,11)))
        self.bin=np.searchsorted(self.edges[1:-1],self.p,side='left')
        self.nb=len(self.edges)-1
        self.brier=(self.p-self.y)**2
        pc=np.clip(self.p,np.finfo(float).eps,1-np.finfo(float).eps)
        self.ll=-(self.y*np.log(pc)+(1-self.y)*np.log1p(-pc))

    def evaluate(self,pw,diagnostics=True):
        w=pw[self.pid].astype(float);n=w.sum();pos=(w*self.y).sum()
        if n<=0:raise ArithmeticError('empty population')
        gt=np.bincount(self.g,weights=w,minlength=self.ng)
        gp=np.bincount(self.g,weights=w*self.y,minlength=self.ng)
        keep=gt>0;gt=gt[keep];gp=gp[keep];s=self.scores[keep]
        rate=pos/n;brier=float(np.sum(w*self.brier)/n)
        out={'events':float(n),'positive_events':float(pos),'prevalence':float(rate),
             'mean_prediction':float(np.sum(w*self.p)/n),'Brier':brier,
             'log_loss':float(np.sum(w*self.ll)/n),
             'observed_expected_ratio':float(pos/np.sum(w*self.p)),
             'Brier_skill_local_prevalence':float(1-brier/(rate*(1-rate))) if 0<rate<1 else np.nan}
        if 0<pos<n:
            # Scores are ascending; half-credit for ties in AUC.
            gn=gt-gp;cneg=np.cumsum(gn)
            out['AUROC']=float(np.sum(gp*(cneg-.5*gn))/(pos*(n-pos)))
            tpr=np.cumsum(gp[::-1]);tot=np.cumsum(gt[::-1])
            out['AP']=float(np.sum(gp[::-1]*(tpr/tot))/pos)
        else:out.update(AUROC=np.nan,AP=np.nan)
        if diagnostics:
            try:out.update(calibration(s,gt,gp));out['calibration_valid']=True;out['calibration_error']=''
            except (ArithmeticError,np.linalg.LinAlgError) as err:
                out.update(calibration_in_the_large=np.nan,calibration_joint_intercept=np.nan,calibration_slope=np.nan,
                           calibration_valid=False,calibration_error=str(err))
        bn=np.bincount(self.bin,weights=w,minlength=self.nb)
        by=np.bincount(self.bin,weights=w*self.y,minlength=self.nb)
        bp=np.bincount(self.bin,weights=w*self.p,minlength=self.nb)
        bo=np.divide(by,bn,out=np.full(self.nb,np.nan),where=bn>0)
        bm=np.divide(bp,bn,out=np.full(self.nb,np.nan),where=bn>0)
        return out,[(float(nn),float(yy),float(pp),float(oo)) for nn,yy,pp,oo in zip(bn,by,bm,bo)]

def verify_math():
    rng=np.random.default_rng(2026100703);maxerr=0.;calerr=0.
    for i in range(18):
        n=180+i*9
        p=np.round(rng.uniform(.02,.95,n),2)
        y=rng.binomial(1,expit(-.45+.85*logit(p)))
        pid=rng.integers(0,63,n);pw=rng.integers(0,4,63)
        f=pd.DataFrame({'y':y,'score':p,'_pid':pid});d=Evaluator(f);r,_=d.evaluate(pw)
        w=pw[pid];keep=w>0
        expected={'AUROC':roc_auc_score(y,p,sample_weight=w),'AP':average_precision_score(y,p,sample_weight=w),
                  'Brier':brier_score_loss(y,p,sample_weight=w),'log_loss':log_loss(y,p,sample_weight=w,labels=[0,1])}
        for key,x in expected.items():maxerr=max(maxerr,abs(x-r[key]));assert abs(x-r[key])<1e-12
        # Independent event-level optimizer and offset root finding.
        z=logit(np.clip(p,1e-6,1-1e-6));X=np.column_stack([np.ones(n),z]);wt=w/w.sum()
        def fun(b):return np.sum(wt*(np.logaddexp(0,X@b)-y*(X@b)))
        def jac(b):return X.T@(wt*(expit(X@b)-y))
        def hess(b):
            q=expit(X@b);return X.T@((wt*q*(1-q))[:,None]*X)
        fit=minimize(fun,[0,1],jac=jac,hess=hess,method='trust-exact',options={'gtol':1e-10})
        assert np.max(abs(jac(fit.x)))<1e-7
        old=np.array([r['calibration_joint_intercept'],r['calibration_slope']])
        err=float(np.max(abs(old-fit.x)));calerr=max(calerr,err);assert err<2e-6
        off=brentq(lambda a:np.sum(w*(y-expit(z+a))),-30,30)
        assert abs(off-r['calibration_in_the_large'])<1e-8
        # Patient multiplicities match physically repeated rows, including ties.
        repeated=np.repeat(np.arange(n),w)
        assert abs(roc_auc_score(y[repeated],p[repeated])-r['AUROC'])<1e-12
        assert abs(average_precision_score(y[repeated],p[repeated])-r['AP'])<1e-12
    return {'status':'PASS','weighted_cases':18,'max_metric_abs_error':maxerr,'max_calibration_abs_error':calerr,
            'zero_weights_tied_scores_repeated_patients':True,'independent_calibration_solver':'scipy trust-exact + offset brentq'}
performance_math_test=verify_math
def first_mask(f,pos):
 idx=f.loc[pos].sort_values(['t0','episode_id'],kind='stable').drop_duplicates('subject_id').index
 a=np.zeros(len(f),bool);a[idx]=True;return a

def summarize(points,boot,keys,metrics):
 rows=[]
 for k,g in boot.groupby(keys,dropna=False,sort=False):
  k=k if isinstance(k,tuple) else (k,);meta=dict(zip(keys,k));ref=points
  for key,v in meta.items():ref=ref[ref[key].eq(v)]
  assert len(ref)==1 and len(g)==1000,(meta,len(ref),len(g))
  for metric in metrics:
   if metric not in g or pd.isna(ref.iloc[0].get(metric)):continue
   a=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy(float);a.sort()
   def q(p):
    if len(a)<950:return np.nan
    z=(len(a)-1)*p;lo=int(np.floor(z));hi=int(np.ceil(z))
    if np.isinf(a[lo]) or (z>lo and np.isinf(a[hi])):return np.inf
    return float(a[lo]+(z-lo)*(a[hi]-a[lo]))
   rows.append(dict(**meta,metric=metric,estimate=ref.iloc[0][metric],low=q(.025),high=q(.975),valid_replicates=len(a)))
 return rows
small=SimpleNamespace(no_hit=no_hit)
base=SimpleNamespace(first_mask=first_mask)
def context(d,pw):
 w=pw[d.pi];return dict(w=w,pw=pw,eligible_events=int(w[d.eligible].sum()),sizes=[int(w[s].sum()) for s in d.strata])

def choose_prefix(order,cumulative,w,k):
 n=int(np.searchsorted(cumulative,k,side='left'))+1 if k else 0;idx=order[:n];take=w[idx].copy()
 if n:take[-1]=k-(int(cumulative[n-2]) if n>1 else 0)
 assert int(take.sum())==k;return idx,take

def fast_value(d,c,strategy,k,selection=None):
 if selection is not None:
  idx,take=selection;ys=d.y[idx];tp=float(take[ys==1].sum());fp=float(take[ys==0].sum());uk=float(take[ys<0].sum());cap=np.zeros(d.np,int)
  np.maximum.at(cap,d.pi[idx[ys==1]],take[ys==1]);new=float(cap[~d.base_pos].sum())
 else:
  ks=[min(k,c['sizes'][0]),max(0,k-c['sizes'][0])];no=np.ones(d.np);tp=fp=uk=0.
  for g,n,t in zip(d.patient_counts,c['sizes'],ks):
   if t:
    tp+=t*np.dot(c['pw'],g['pos'])/n;fp+=t*np.dot(c['pw'],g['neg'])/n;uk+=t*np.dot(c['pw'],g['unknown'])/n
   no*=small.no_hit(n,t,g['pos'])
  new=float(np.dot(c['pw'][~d.base_pos],(1-no)[~d.base_pos]))
 first_h90=float(c['w'][d.first&d.hist].sum());first_total=float(c['w'][d.first].sum())
 if selection is not None:first_add=float(take[d.first[idx]].sum())
 else:first_add=sum(t*float(c['w'][d.first&st].sum())/n if n else 0 for st,n,t in zip(d.strata,c['sizes'],ks))
 baseline=int(c['pw'][d.base_pos].sum());assert np.isclose(tp+fp+uk,k,atol=1e-8)
 return dict(first_h90=first_h90,first_total=first_total,first_add=first_add,first_combined=first_h90+first_add,first_coverage=(first_h90+first_add)/first_total if first_total else np.nan,added_alerts=k,positive_events=tp,known_nonpositive_events=fp,unknown_events=uk,new_positive_patients=new,baseline_positive_patients=baseline,union_positive_patients=baseline+new,reviews_per_new_positive_patient=k/new if new else np.inf,eligible_events=c['eligible_events'])

def calculate(ds,pw):
 vals={};out=[];first=next(iter(ds.values()));c=context(first,pw);cumulative={}
 for name,d in ds.items():
  assert np.array_equal(d.pi,first.pi) and np.array_equal(d.hist,first.hist)
  cumulative[name]={s:np.cumsum(c['w'][o]) for s,o in d.orders.items()}
  for pct in BUDGETS:
   k=(pct*c['eligible_events']+99)//100
   for strategy in STRATEGIES:
    selection=choose_prefix(d.orders[strategy],cumulative[name][strategy],c['w'],k) if strategy in d.orders else None
    v=fast_value(d,c,strategy,k,selection);vals[name,pct,strategy]=v;out.append(dict(variant=name,budget_percent=pct,strategy=strategy,**v))
   for comp in STRATEGIES[1:]:
    a=vals[name,pct,'model'];b=vals[name,pct,comp]
    out.append(dict(variant=name,budget_percent=pct,strategy='model_minus_'+comp,added_alerts=k,**{key:a[key]-b[key] for key in CM if key!='reviews_per_new_positive_patient'}))
 ref=next(iter(ds))
 for name in list(ds)[1:]:
  for pct in BUDGETS:
   for strategy in ['model','model_minus_icu_expected','model_minus_icu_los']:
    a=next(x for x in out if x['variant']==name and x['budget_percent']==pct and x['strategy']==strategy);b=next(x for x in out if x['variant']==ref and x['budget_percent']==pct and x['strategy']==strategy)
    out.append(dict(variant=name+'_minus_'+ref,budget_percent=pct,strategy=strategy,**{key:a[key]-b[key] for key in CM if key!='reviews_per_new_positive_patient'}))
 return out
def pipeline(plan, features, C):
    assert set(features) <= set(plan['features']) and not set(features) & set(plan['never_predictors'])
    numeric = [f for f in features if f in plan['numeric_features']]
    categorical = [f for f in features if f not in numeric]
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                              ('scale', StandardScaler())]), numeric),
        ('category', OneHotEncoder(categories=[plan['categorical_domains'][f] for f in categorical],
                                  sparse_output=False, handle_unknown='error'), categorical),
    ], remainder='drop')
    opts = plan['learner']
    return Pipeline([('preprocess', prep), ('model', LogisticRegression(
        C=C, penalty='l2', solver=opts['solver'], max_iter=opts['max_iter'], tol=opts['tol'],
        fit_intercept=opts['fit_intercept'], class_weight=None, random_state=opts['random_state']))])
def uniform_ties(f):return np.array([int(hashlib.sha256(('adult_workflow_v1|'+str(e)).encode()).hexdigest()[:13],16) for e in f.episode_id],dtype=np.int64)

def candidate(d,tie,row):
 if row['strategy']=='D18':
  cut=row['score_cutoff'];flag=(d.score>cut)|((d.score==cut)&(tie<=row['tie_cutoff']))
 else:
  los=np.nan_to_num(d.los,nan=-np.inf);i=row['icu_cutoff'];l=row['los_cutoff'];flag=(d.icu.astype(int)>i)|((d.icu.astype(int)==i)&((los>l)|((los==l)&(tie<=row['tie_cutoff']))))
 return flag&d.eligible

def cooldown(d,candidates,days):
 if not days:return candidates.copy()
 indices=np.flatnonzero(candidates);times=pd.DatetimeIndex(pd.to_datetime(d.f.t0,utc=True)).as_unit('ns').asi8//1000000000
 order=indices[np.lexsort((d.f.episode_id.to_numpy()[indices],times[indices],d.pi[indices]))]
 emitted=np.zeros(len(d.f),bool);last=np.full(d.np,-10**18,dtype=np.int64);delta=days*86400
 for j in order:
  p=d.pi[j]
  if times[j]-last[p]>=delta:emitted[j]=True;last[p]=times[j]
 return emitted

def patient_counts(d,mask):
 count=lambda sel:np.bincount(d.pi[sel],minlength=d.np)
 pc=count(mask&d.pos);first=base.first_mask(d.f,d.pos)
 return np.column_stack([count(mask),pc,count(mask&d.neg),count(mask&(d.y<0)),pc>0,(pc>0)&~d.base_pos,count(mask&first),count(mask)>0]).astype(float)
