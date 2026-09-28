from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
from q1.quality_models import QUALITY_FIELDS
from q4.capability import BENCH_COLS

DOM17=['arxiv','freelaw','nih_exporter','pubmed_central','wikipedia_en','dm_mathematics','github','philpapers','stackexchange','enron_emails','gutenberg_pg_19','pile_cc','ubuntu_irc','europarl','hackernews','pubmed_abstracts','uspto_backgrounds']
QDOM=['arxiv','book','c4','commoncrawl','github','stackexchange','wikipedia']

def main(root=Path('tests/synthetic_processed'),seed=42):
 rng=np.random.default_rng(seed);root=Path(root);(root/'A').mkdir(parents=True,exist_ok=True);(root/'B').mkdir(parents=True,exist_ok=True);(root/'C').mkdir(parents=True,exist_ok=True)
 # Q1 quality
 rows=[]
 for src,n,doms in [('A1',700,QDOM),('A2',300,['arxiv']),('A3',500,['github'])]:
  for i in range(n):
   d=rng.choice(doms);base=rng.beta(5,2);r={'record_id':f'{src}_{i}','source_set':src,'source_file':f'{src}.xz','quality_domain':d,'sub_path':''}
   for c in QUALITY_FIELDS:
    val=np.clip(base+rng.normal(0,.12),0,1)
    if c in ['ad_en','rps_doc_frac_no_alph_words','rps_doc_frac_chars_top_2gram','rps_doc_frac_chars_top_3gram','rps_lines_uppercase_letter_fraction','rps_lines_numerical_chars_fraction']:val=1-val
    r[c]=val
   rows.append(r)
 pd.DataFrame(rows).to_csv(root/'A'/'q1_quality.csv',index=False)
 # mixtures
 pcols=['p_'+d for d in DOM17]; lossdom=[d for d in DOM17 if d not in ['nih_exporter','enron_emails','europarl','philpapers']]; lcols=['loss_'+d for d in lossdom]
 beta=rng.normal(2.5,.2,17); gam=rng.normal(0,.12,(17,17));gam=(gam+gam.T)/2;np.fill_diagonal(gam,0)
 pairs=[('A4_A5','train','1M',180,0.0),('A6_A7','test','1M',90,0.0),('A8_A9','test','60M',90,-.35),('A10_A11','test','1B',60,-.7),('A12_A13','extrapolation','10B',50,-1.0),('A14_A15','extrapolation','70B',50,-1.25)]
 mrows=[]
 for pid,split,scale,n,shift in pairs:
  for i in range(n):
   p=rng.dirichlet(np.ones(17)*1.5);mix=beta@p+sum(gam[a,b]*p[a]*p[b] for a in range(17) for b in range(a+1,17));r={'index':i,'pair_id':pid,'split':split,'model_scale_label':scale,'model_scale_B':{'1M':.001,'60M':.06,'1B':1,'10B':10,'70B':70}[scale],'mixture_nature':'real','loss_nature':'real'};r.update(dict(zip(pcols,p)))
   for k,c in enumerate(lcols):r[c]=mix+shift+rng.normal(0,.03)+.03*k
   mrows.append(r)
 pd.DataFrame(mrows).to_csv(root/'A'/'q1_mixture_loss.csv',index=False)
 mapping=[]
 for i,d in enumerate(DOM17):mapping.append({'mixture_domain':d,'quality_domain':QDOM[i%len(QDOM)],'mapping_type':'synthetic','domain_sample_rows':100,'domain_avg_text_chars':1000,'domain_sample_bytes_requested':1e6,'domain_source_path':'x'})
 pd.DataFrame(mapping).to_csv(root/'A'/'q1_domain_mapping.csv',index=False)
 pd.DataFrame({'row_no':range(200),'domain':rng.choice(DOM17,200),'text_char_count':rng.integers(100,5000,200),'text_line_count':rng.integers(1,100,200),'text_whitespace_token_count':rng.integers(20,1000,200)}).to_csv(root/'A'/'q1_text_validation.csv',index=False)
 # Q2 scaling
 E,A,B,alpha,beta0=1.2,.8,1.1,.25,.28
 sr=[]
 for sid,nrow,nature,role in [('B1',160,'real','main_fit'),('B2',80,'semi_synthetic','family_out_validation'),('B3',100,'interpolated','trajectory_validation'),('B4',50,'real','cross_family_validation'),('B5',40,'real','literature_validation'),('B6',120,'semi_synthetic','quality_basic'),('B7',120,'semi_synthetic','quality_expanded'),('B8',160,'semi_synthetic','quality_large'),('B10',60,'estimated','large_model_extrapolation')]:
  for i in range(nrow):
   N=10**rng.uniform(-1,2.3);D=10**rng.uniform(-1,3);q=rng.uniform(.4,1.0) if sid in ['B6','B7','B8'] else np.nan;loss=E+A*N**(-alpha)+B*D**(-beta0)
   if np.isfinite(q):loss=E+A*N**(-alpha)+B*(D*np.exp(1.2*(q-1)))**(-beta0)
   loss+=rng.normal(0,.015);sr.append({'source_id':sid,'data_nature':nature,'role':role,'source_file':sid+'.csv','record_id':f'{sid}_{i//20}','model_family':'pythia' if sid!='B2' else 'cerebras','N_params_B':N,'D_tokens_B':D,'C_FLOPs_1e21':6*N*D/1e3,'N_params':N*1e9,'D_tokens':D*1e9,'C_FLOPs':6*N*D*1e18,'C_6ND_FLOPs':6*N*D*1e18,'C_to_6ND_ratio':1,'val_loss':loss,'Q_score':q,'step':i,'steps':nrow,'is_converged':True,'data_type':'synthetic','family_key':'pythia','family_metadata_available':True})
 pd.DataFrame(sr).to_csv(root/'B'/'q2_scaling.csv',index=False)
 lm=[]
 for i in range(50):
  N=10**rng.uniform(2,4);D=10**rng.uniform(2,4);lm.append({'model_name':f'large{i}','model_key':f'large{i}','N_params_B':N,'D_tokens_B':D,'FLOPs':6*N*D*1e18,'publication_date':f'2025-01-{(i%28)+1:02d}','organization':'org','accessibility':'open','country':'x','N_params':N*1e9,'D_tokens':D*1e9,'C_6ND_FLOPs':6*N*D*1e18})
 pd.DataFrame(lm).to_csv(root/'B'/'q2_large_models.csv',index=False)
 # Q3 context
 pd.DataFrame([{'model_name':f'm{i}','model_key':f'm{i}','n_layers':24,'n_heads':16,'d_model':2048,'vocab_size':50000,'max_position_embeddings':v,'training_data_TB':1} for i,v in enumerate([2048,4096,8192,32768,131072])]).to_csv(root/'C'/'q3_context.csv',index=False)
 # Q4 leaderboard/epoch/timeseries
 lbs=[];eps=[];tss=[]
 start=pd.Timestamp('2019-01-01')
 for i in range(100):
  date=start+pd.DateOffset(days=int(i*25));N=10**rng.uniform(8,11);D=10**rng.uniform(10,13);C=6*N*D*rng.uniform(.8,1.2);trend=i/100
  vals=np.clip(20+65*trend+rng.normal(0,7,6),0,100);model=f'model{i}';typ='pretrained' if i%3==0 else 'posttrained'
  r={'model':model,'model_key':model,'params_B':N/1e9,'submission_date':date,'hub_license':'apache-2.0','type_raw':typ,'model_type_bucket':typ,'average':vals.mean(),'params':N,'epoch_publication_date':date,'epoch_organization':'org','epoch_open_weights':True,'benchmark_complete':True};r.update(dict(zip(BENCH_COLS,vals)));lbs.append(r)
  eps.append({'model':model,'model_key':model,'domain':'Language','task':'LM','organization':'org','publication_date':date,'parameters':N,'training_compute_flop':C,'training_dataset_size_total':D,'approach':'transformer','confidence':'high','model_accessibility':'open','base_model':'','finetune_compute_flop':np.nan,'post_training_compute_flop':np.nan,'frontier_model':False,'open_model_weights':'Yes','parameters_B':N/1e9,'open_weights_bool':True})
  tss.append({'model':model,'model_key':model,'year':date.year,'params_B':N/1e9,'average':vals.mean(),**dict(zip(BENCH_COLS,vals)),'source':'synthetic','params':N})
 pd.DataFrame(lbs).to_csv(root/'C'/'q4_leaderboard.csv',index=False);pd.DataFrame(eps).to_csv(root/'C'/'q4_epoch_models.csv',index=False);pd.DataFrame(tss).to_csv(root/'C'/'q4_timeseries.csv',index=False)
 # bridge
 br=[]
 for i in range(60):
  loss=rng.uniform(1.5,4.0);base=100-22*loss
  vals=np.clip(base+rng.normal(0,4,6),0,100);br.append({'model':f'b{i}','model_key':f'b{i}','N_params_B':1,'D_tokens_B':1,'val_loss':loss,'benchmark_average':vals.mean(),**dict(zip(BENCH_COLS,vals)),'loss_source':'synthetic','loss_comparability':'High' if i<35 else 'Medium','N_params':1e9,'D_tokens':1e9,'comparability_bucket':'high' if i<35 else 'medium'})
 pd.DataFrame(br).to_csv(root/'C'/'q4_loss_benchmark_bridge.csv',index=False)
 # task metrics
 tr=[]
 metrics={'IFEval':'prompt_level_strict_acc,none','BBH':'acc_norm,none','MATH':'exact_match,none','GPQA':'acc_norm,none','MUSR':'acc_norm,none','MMLU_PRO':'acc,none'}
 for i in range(80):
  for g,m in metrics.items():
   for t in range(3):tr.append({'model_name':f'model{i}','model_key':f'model{i}','model_dir':f'model{i}','selected_file':'x.json','evaluation_date':'2025-01-01','task':f'{g.lower()}_{t}','task_group':g,'is_aggregate_task':False,'metric':m,'value':rng.uniform(.2,.9),'higher_is_better':True,'n_samples_original':100,'n_samples_effective':100})
 pd.DataFrame(tr).to_csv(root/'C'/'q4_task_metrics.csv',index=False)
 print(root)

if __name__=='__main__':main()
