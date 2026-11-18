from __future__ import annotations
import io,json,math,time,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd,requests
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT=Path('research_outputs/fluid_network');OUT.mkdir(parents=True,exist_ok=True)
EPS=1e-12;RNG=np.random.default_rng(20261118)
SYMS=['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT','DOGEUSDT','LTCUSDT','BCHUSDT','LINKUSDT','ETCUSDT','XLMUSDT','DOTUSDT','SOLUSDT']
H=[4,12,24,72,120,480]
BASE=['u','m4','m12','m24','m72','rv24','rv72','qz','imb6','imb24','range24','peer']
CANDS=['advective_gap','pressure_gap','continuity_sink','subgrid_flux','circulation','reattachment','consensus']

def z(s,w=720,m=240):
 p=s.shift(1);return ((s-p.rolling(w,m).mean())/(p.rolling(w,m).std(ddof=0)+EPS)).clip(-12,12)
def mom(s,w):
 return (s.rolling(w,w).sum()/np.sqrt(s.pow(2).rolling(w,w).sum()+EPS)).clip(-12,12)
def dl(sym,per):
 u=f'https://data.binance.vision/data/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{per}.zip'
 for a in range(4):
  try:
   r=requests.get(u,timeout=40,headers={'User-Agent':'fluid-network-study'});
   if r.status_code==404:return None
   r.raise_for_status()
   with zipfile.ZipFile(io.BytesIO(r.content)) as q:
    n=[x for x in q.namelist() if x.endswith('.csv')][0];x=pd.read_csv(q.open(n),header=None,low_memory=False).iloc[:,:12]
   x.columns=['ot','open','high','low','close','volume','ct','qv','trades','tb','tbq','ig']
   x['ot']=pd.to_numeric(x['ot'],errors='coerce');x=x[x.ot.notna()].copy()
   for c in ['open','high','low','close','qv','trades','tbq']:x[c]=pd.to_numeric(x[c],errors='coerce')
   x['time']=pd.to_datetime(x.ot.astype('int64'),unit='ms',utc=True);x['symbol']=sym
   return x[['time','symbol','open','high','low','close','qv','trades','tbq']]
  except Exception:
   time.sleep(a+1)
 return None

def load():
 periods=[str(p) for p in pd.period_range('2021-01','2026-10',freq='M')];fs=[]
 with ThreadPoolExecutor(max_workers=16) as ex:
  fut=[ex.submit(dl,s,p) for s in SYMS for p in periods]
  for i,f in enumerate(as_completed(fut),1):
   x=f.result()
   if x is not None and len(x):fs.append(x)
   if i%100==0:print('download',i,len(fut),len(fs),flush=True)
 d=pd.concat(fs,ignore_index=True).drop_duplicates(['time','symbol']).sort_values(['time','symbol'])
 cnt=d.groupby('symbol').size();keep=cnt[cnt>=40000].index.tolist();d=d[d.symbol.isin(keep)]
 return d,keep

def wide(d,c,sy):return d.pivot(index='time',columns='symbol',values=c).sort_index().reindex(columns=sy)
def csic(x,y):
 out=[]
 for t in x.index:
  a=x.loc[t].to_numpy();b=y.loc[t].to_numpy();g=np.isfinite(a)&np.isfinite(b)
  out.append(spearmanr(a[g],b[g]).statistic if g.sum()>=8 else np.nan)
 return pd.Series(out,index=x.index)
def boot(s,rep=1200,block=14):
 a=s.dropna().resample('1D').mean().dropna().to_numpy();n=len(a);z=[]
 for _ in range(rep):
  q=[]
  while len(q)<n:
   k=int(RNG.integers(n));q.extend(a[(k+np.arange(block))%n])
  z.append(np.mean(q[:n]))
 return {'mean':float(np.mean(a)),'lo':float(np.quantile(z,.025)),'hi':float(np.quantile(z,.975)),'p':float(np.mean(np.array(z)<=0))}
def holm(p):
 q=sorted(p.items(),key=lambda x:x[1]);m=len(q);o={};r=0
 for i,(k,v) in enumerate(q):r=max(r,min(1,(m-i)*v));o[k]=r
 return o

def model_inc(train,test,fac,y):
 cols=BASE;tr=train.dropna(subset=cols+[fac,y]);te=test.dropna(subset=cols+[fac,y])
 sc0=StandardScaler().fit(tr[cols]);sc1=StandardScaler().fit(tr[cols+[fac]])
 m0=Ridge(alpha=20).fit(sc0.transform(tr[cols]),tr[y]);m1=Ridge(alpha=20).fit(sc1.transform(tr[cols+[fac]]),tr[y])
 p0=m0.predict(sc0.transform(te[cols]));p1=m1.predict(sc1.transform(te[cols+[fac]]));yy=te[y].to_numpy()
 return 100*(np.mean((yy-p0)**2)-np.mean((yy-p1)**2))/(np.mean((yy-p0)**2)+EPS)
def residual_ic(train,test,fac,y):
 tr=train.dropna(subset=BASE+[fac]);te=test.dropna(subset=BASE+[fac,y]);sc=StandardScaler().fit(tr[BASE]);m=Ridge(alpha=20).fit(sc.transform(tr[BASE]),tr[fac]);te=te.copy();te['res']=te[fac]-m.predict(sc.transform(te[BASE]));
 return float(te.groupby('time').apply(lambda g:spearmanr(g.res,g[y]).statistic if len(g)>=8 else np.nan,include_groups=False).mean())
def port(df,fac,h,cost=2):
 x=df.pivot(index='time',columns='symbol',values=fac);r=df.pivot(index='time',columns='symbol',values='nextret');rk=x.rank(axis=1,pct=True);w=pd.DataFrame(0.,index=x.index,columns=x.columns);w[rk>=.75]=1;w[rk<=.25]=-1;w=w.div(w.abs().sum(axis=1).replace(0,np.nan),axis=0).fillna(0);w=w.shift(1).rolling(h,1).mean();gross=(w*r).sum(axis=1);to=w.diff().abs().sum(axis=1);p=gross-to*cost*1e-4;ann=24*365;v=p.std(ddof=0)*math.sqrt(ann);return {'ret':float(p.mean()*ann),'sharpe':float(p.mean()*ann/(v+EPS)),'turnover':float(to.mean())}

def main():
 d,sy=load();F={c:wide(d,c,sy) for c in ['open','high','low','close','qv','trades','tbq']};mask=F['close'].notna().all(axis=1);F={k:v.loc[mask] for k,v in F.items()};c=F['close'];r=np.log(c).diff();scale=np.sqrt(r.shift(1).pow(2).ewm(span=48,min_periods=24,adjust=False).mean());u=(r/(scale+EPS)).clip(-12,12);market=u.median(axis=1);dev=u.index<pd.Timestamp('2023-01-01',tz='UTC');beta=u.loc[dev].apply(lambda s:s.cov(market.loc[dev])/(market.loc[dev].var()+EPS));v=u-market.to_numpy()[:,None]*beta.to_numpy()[None,:]
 qz=np.log1p(F['qv'].clip(lower=0)).apply(lambda s:z(s,168,72));rho=np.exp(qz.clip(-3,3));imb=(2*F['tbq']/(F['qv']+EPS)-1).clip(-1,1);rng=np.log(F['high']/F['low']).clip(lower=0)
 feat={'u':u,'m4':u.apply(lambda s:mom(s,4)),'m12':u.apply(lambda s:mom(s,12)),'m24':u.apply(lambda s:mom(s,24)),'m72':u.apply(lambda s:mom(s,72)),'rv24':np.log(np.sqrt(u.pow(2).rolling(24,18).mean())+EPS).apply(lambda s:z(s,336,120)),'rv72':np.log(np.sqrt(u.pow(2).rolling(72,54).mean())+EPS).apply(lambda s:z(s,336,120)),'qz':qz,'imb6':imb.rolling(6,4).mean().apply(lambda s:z(s,336,120)),'imb24':imb.rolling(24,18).mean().apply(lambda s:z(s,336,120)),'range24':rng.rolling(24,18).mean().apply(lambda s:z(s,336,120))}
 dv=v.loc[dev].dropna();corr=dv.corr().fillna(0).to_numpy();np.fill_diagonal(corr,0);W=np.zeros_like(corr)
 for i in range(len(sy)):
  j=np.argsort(np.maximum(corr[i],0))[-3:];W[i,j]=np.maximum(corr[i,j],0)
 W=(W+W.T)/2;W=W/(W.sum(1,keepdims=True)+EPS);L=np.eye(len(sy))-W;P=np.linalg.pinv(L+.15*np.eye(len(sy)))
 xx=dv.shift(1).dropna();yy=dv.loc[xx.index];reg=Ridge(alpha=25).fit(xx,yy);B=reg.coef_;np.fill_diagonal(B,0)
 arr=dv.to_numpy();A=np.zeros_like(W)
 for i in range(len(sy)):
  for j in range(len(sy)):
   if i!=j:A[i,j]=np.corrcoef(arr[1:,i],arr[:-1,j])[0,1]-np.corrcoef(arr[1:,j],arr[:-1,i])[0,1]
 A=A/(np.abs(A).sum(1,keepdims=True)+EPS)
 va=v.to_numpy();ra=rho.to_numpy();coarse=va@W.T;feat['peer']=pd.DataFrame(coarse,index=v.index,columns=sy);adv=va@B.T-va;div=np.full_like(va,np.nan)
 for t in range(len(va)):
  if np.isfinite(va[t]).all() and np.isfinite(ra[t]).all():div[t]=(W*(.5*(ra[t,:,None]+ra[t,None,:]))*(va[t,None,:]-va[t,:,None])).sum(1)
 ctr=div-np.nanmean(div,1,keepdims=True);press=np.full_like(ctr,np.nan);g=np.isfinite(ctr).all(1);press[g]=-ctr[g]@P.T;cont=-(np.log(rho+EPS).diff().to_numpy()+div);tau=(va*va)@W.T-coarse*coarse;sg=tau*(coarse-va);circ=(ra*va)@A.T;sep=np.abs(va-coarse);sp=np.roll(sep,1,0);sp[0]=np.nan;reat=-va*sp*np.maximum(sp-sep,0)*np.exp(-np.abs(feat['range24'].to_numpy())/2)
 cand={'advective_gap':adv,'pressure_gap':press,'continuity_sink':cont,'subgrid_flux':sg,'circulation':circ,'reattachment':reat};C={k:pd.DataFrame(x,index=v.index,columns=sy).apply(z) for k,x in cand.items()};C['consensus']=sum([(C[k].rank(axis=1,pct=True)-.5)*(-1 if k=='subgrid_flux' else 1) for k in cand])/len(cand)
 labels={h:np.log(c.shift(-h-1)/c.shift(-1))/(scale*math.sqrt(h)+EPS) for h in H};rows=[]
 for name,x in {**feat,**C}.items():rows.append(x.stack(dropna=False).rename(name))
 panel=pd.concat(rows,axis=1).reset_index(names=['time','symbol'])
 for h,y in labels.items():panel=panel.merge(y.stack(dropna=False).rename(f'y{h}').reset_index(),on=['time','symbol'],how='left')
 panel=panel.merge(r.shift(-1).stack(dropna=False).rename('nextret').reset_index(),on=['time','symbol'],how='left');D=panel[panel.time<pd.Timestamp('2023-01-01',tz='UTC')];V=panel[(panel.time>=pd.Timestamp('2023-01-01',tz='UTC'))&(panel.time<pd.Timestamp('2025-01-01',tz='UTC'))];O=panel[panel.time>=pd.Timestamp('2025-01-01',tz='UTC')]
 sel={};scan=[]
 for f in CANDS:
  best=(-9,None,None)
  for h in H:
   ic=float(csic(C[f].loc[D.time.min():D.time.max()],labels[h].loc[D.time.min():D.time.max()]).mean())
   for s in [1,-1]:
    scan.append({'family':f,'h':h,'sign':s,'dev_ic':s*ic})
    if s*ic>best[0]:best=(s*ic,h,s)
  sel[f]={'h':best[1],'sign':best[2],'dev_ic':best[0]};D=D.copy();V=V.copy();O=O.copy();D['F_'+f]=D[f]*best[2];V['F_'+f]=V[f]*best[2];O['F_'+f]=O[f]*best[2]
 pd.DataFrame(scan).to_csv(OUT/'development_scan.csv',index=False);res=[];ps={}
 for f,s in sel.items():
  y='y'+str(s['h']);fac='F_'+f;series=V.groupby('time').apply(lambda q:spearmanr(q[fac],q[y],nan_policy='omit').statistic if q[[fac,y]].dropna().shape[0]>=8 else np.nan,include_groups=False);b=boot(series);ps[f]=b['p'];rec={'family':f,**s,'val_ic':float(series.mean()),**b,'mse_gain':model_inc(D,V,fac,y),'resid_ic':residual_ic(D,V,fac,y),**port(V,fac,s['h'])};res.append(rec)
 adj=holm(ps)
 for q in res:q['holm_p']=adj[q['family']];q['pass']=q['val_ic']>.015 and q['holm_p']<=.1 and q['mse_gain']>0 and q['resid_ic']>.005 and q['ret']>0 and q['sharpe']>0
 pd.DataFrame(res).to_csv(OUT/'validation.csv',index=False);hold=[];ad=[]
 for q in res:
  if not q['pass']:continue
  f=q['family'];fac='F_'+f;y='y'+str(q['h']);series=O.groupby('time').apply(lambda zq:spearmanr(zq[fac],zq[y],nan_policy='omit').statistic if zq[[fac,y]].dropna().shape[0]>=8 else np.nan,include_groups=False);qq={'family':f,'h':q['h'],'hold_ic':float(series.mean()),'mse_gain':model_inc(pd.concat([D,V]),O,fac,y),'resid_ic':residual_ic(pd.concat([D,V]),O,fac,y),**port(O,fac,q['h'])};qq['pass']=qq['hold_ic']>0 and qq['mse_gain']>=0 and qq['resid_ic']>0 and qq['ret']>0 and qq['sharpe']>0;hold.append(qq)
  if qq['pass']:ad.append({'family':f,'spec':sel[f],'validation':q,'holdout':qq})
 pd.DataFrame(hold).to_csv(OUT/'holdout.csv',index=False);summary={'symbols':sy,'rows':len(d),'min':str(d.time.min()),'max':str(d.time.max()),'selection':sel,'validation':res,'holdout':hold,'admitted':ad,'admitted_count':len(ad),'W':W.tolist(),'B':B.tolist(),'A':A.tolist()};(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str));print(json.dumps({'admitted_count':len(ad),'admitted':[x['family'] for x in ad]},indent=2))
if __name__=='__main__':main()
