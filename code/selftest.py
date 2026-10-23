from runtime import *
import argparse

def main():
 expected=pd.read_csv(ROOT/'examples/expected_scores.csv',float_precision='round_trip')
 x=pd.read_csv(ROOT/'examples/synthetic_18_features.csv',dtype=str,keep_default_na=False)
 checked=0;errors={}
 for key,entry in REGISTRY.items():
  for item in [entry]+entry.get('outer_models',[]):
   assert sha(ROOT/item['file'])==item['sha256'],item['file'];checked+=1
  err=float(np.max(np.abs(predict(x,key)-expected[key].to_numpy(float))));assert err<1e-12,(key,err);errors[key]=err
 kernels.allocation_math_test();kernels.performance_math_test()
 eng=x.copy();eng['sex']=eng.sex.map({'男':'male','女':'Female','未知':'UNKNOWN'}).fillna(eng.sex)
 for key in REGISTRY:assert np.array_equal(predict(eng,key),predict(x,key)),('English sex labels changed predictions',key)
 for field,value in [('age','17'),('prior_crgn_reported_90d','0'),('length_of_stay','-1')]:
  bad=x.copy();bad.loc[0,field]=value
  try:prepare_features(bad)
  except ValueError:pass
  else:raise AssertionError('Failed to reject '+field)
 verified=0
 if (ROOT/'PACKAGE_MANIFEST.json').exists():
  for rel,item in read(ROOT/'PACKAGE_MANIFEST.json')['files'].items():
   assert sha(ROOT/rel)==item['sha256'],rel;verified+=1
 print(json.dumps(dict(status='PASS',frozen_model_files=checked,synthetic_examples=len(x),prediction_max_abs_error=errors,math_checks=True,input_boundary_checks=True,english_sex_label_check=True,package_files_verified=verified),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
