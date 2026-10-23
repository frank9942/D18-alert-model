"""Score a prepared adult 18-feature CSV; generates scores, not treatment orders."""
from runtime import *
import argparse

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--model',choices=list(REGISTRY),default='D18');p.add_argument('--outer-fold',type=int,choices=range(5));a=p.parse_args()
 out=Path(a.out)
 if out.exists():raise FileExistsError('Use a new output filename: '+str(out))
 raw=pd.read_csv(a.input,dtype=str,keep_default_na=False);scores=predict(raw,a.model,a.outer_fold)
 cols=[x for x in ['episode_id','subject_id','t0','y','icu_status','length_of_stay','prior_crgn_reported_90d'] if x in raw]
 result=raw[cols].copy();result['score']=scores;result['H90_flag']=raw.prior_crgn_reported_90d.eq('1');result['score_role']='nested_outer_fold' if a.outer_fold is not None else 'frozen_final_model'
 out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,index=False,encoding='utf-8-sig');print('SCORED',len(result),a.model,str(out))
if __name__=='__main__':main()
