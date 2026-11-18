from __future__ import annotations
import json, math, random, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SEED=20261118; EPS=1e-12; random.seed(SEED); RNG=np.random.default_rng(SEED)
OUT=Path('research_outputs/ashare_fluid_round10'); OUT.mkdir(parents=True,exist_ok=True)
DEV0,DEV1=pd.Timestamp('2019-01-01'),pd.Timestamp('2022-12-31')
VAL0,VAL1=pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')
LAT0,LAT1=pd.Timestamp('2025-01-01'),pd.Timestamp('2026-11-15')
HEAD={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'}
QUOTE='https://82.push2.eastmoney.com/api/qt/clist/get'; HIST='https://push2his.eastmoney.com/api/qt/stock/kline/get'

def fnum(x):
    try:return float(x)
    except:return np.nan

def universe(n_each=250):
    p={'pn':1,'pz':6000,'po':1,'np':1,'ut':'bd1d9ddb04089700cf9c27f6f7426281','fltt':2,'invt':2,'fid':'f6','fs':'m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23','fields':'f12,f14,f6,f21'}
    err=''
    for k in range(6):
        try:
            r=requests.get(QUOTE,params=p,headers=HEAD,timeout=30); r.raise_for_status(); d=r.json()['data']['diff']
            if isinstance(d,dict):d=list(d.values())
            rows=[]
            for z in d:
                c=str(z.get('f12','')); name=str(z.get('f14',''))
                if len(c)!=6 or any(q in name.upper() for q in ('ST','退','N ','C ')):continue
                if c.startswith('6'):m,sec='SH','1.'+c
                elif c.startswith(('0','3')):m,sec='SZ','0.'+c
                else:continue
                rows.append((c,name,m,sec,fnum(z.get('f6')),fnum(z.get('f21'))))
            x=pd.DataFrame(rows,columns=['code','name','market','secid','spot_amount','float_cap'])
            if len(x)<1000:raise RuntimeError(f'small universe {len(x)}')
            x['sel']=x.spot_amount.where(x.spot_amount>0,x.float_cap)
            return x.sort_values(['market','sel'],ascending=[True,False]).groupby('market').head(n_each).reset_index(drop=True)
        except Exception as e:err=repr(e);time.sleep(2+k)
    raise RuntimeError(err)

def one(row):
    p={'secid':row['secid'],'ut':'fa5fd1943c7b386f172d6893dbfba10b','fields1':'f1,f2,f3,f4,f5,f6','fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61','klt':101,'fqt':1,'beg':'20180101','end':'20261115','lmt':1000000}
    err=''
    for k in range(5):
        try:
            r=requests.get(HIST,params=p,headers=HEAD,timeout=35);r.raise_for_status();d=r.json().get('data');kl=[] if not d else d.get('klines',[])
            if not kl:return row['code'],None,'empty'
            a=[s.split(',') for s in kl]
            x=pd.DataFrame(a,columns=['date','open','close','high','low','volume','amount','amplitude','pct','change','turnover'])
            x.date=pd.to_datetime(x.date,errors='coerce')
            for c in x.columns[1:]:x[c]=pd.to_numeric(x[c],errors='coerce')
            x['code']=row['code'];x['market']=row['market'];x['name']=row['name']
            x=x.replace([np.inf,-np.inf],np.nan).dropna(subset=['date','open','close','high','low','amount'])
            x=x[(x[['open','close','high','low']]>0).all(axis=1)&(x.amount>0)]
            return (row['code'],x,None) if len(x)>=900 else (row['code'],None,f'short_{len(x)}')
        except Exception as e:err=repr(e);time.sleep(1.5*(k+1))
    return row['code'],None,err

def load():
    u=universe(); frames=[];errors=[];t=time.time()
    with ThreadPoolExecutor(max_workers=20) as ex:
        fs=[ex.submit(one,r) for r in u.to_dict('records')]
        for i,q in enumerate(as_completed(fs),1):
            c,x,e=q.result();frames.append(x) if x is not None else errors.append({'code':c,'error':e})
            if i%50==0:print('download',i,len(frames),'elapsed',round(time.time()-t,1),flush=True)
    if not frames:raise RuntimeError('no data')
    x=pd.concat(frames,ignore_index=True).drop_duplicates(['code','date']).sort_values(['code','date']).reset_index(drop=True)
    by=x.groupby('market').code.nunique().to_dict()
    if min(by.values())<100:raise RuntimeError(str(by))
    return x,{'requested':len(u),'usable':x.code.nunique(),'by_market':by,'rows':len(x),'min_date':str(x.date.min().date()),'max_date':str(x.date.max().date()),'errors':errors[:100],'source':'Eastmoney public adjusted daily OHLCV','selection':'current non-ST top activity, equal SH/SZ caps','survivorship_bias':True}

def prior_z(s,code,w=120,m=60):
    p=s.groupby(code,sort=False).shift(1);g=p.groupby(code,sort=False);mu=g.transform(lambda a:a.rolling(w,min_periods=m).mean());sd=g.transform(lambda a:a.rolling(w,min_periods=m).std(ddof=0));return ((s-mu)/(sd+EPS)).clip(-10,10)

def age_max(a,w):
    a=np.asarray(a,float);o=np.full(len(a),np.nan);q=deque()
    for i,v in enumerate(a):
        z=v if np.isfinite(v) else -np.inf
        while q and (a[q[-1]] if np.isfinite(a[q[-1]]) else -np.inf)<=z:q.pop()
        q.append(i)
        while q and q[0]<=i-w:q.popleft()
        if i>=w-1 and q and np.isfinite(a[q[0]]):o[i]=i-q[0]
    return o

def cs_z(df,c):
    def z(s):
        lo,hi=s.quantile(.01),s.quantile(.99);a=s.clip(lo,hi);return (a-a.mean())/(a.std(ddof=0)+EPS)
    return df.groupby('date',sort=False)[c].transform(z)

def rankic(df,score,target='target5',keys=['date'],minn=20):
    u=df[keys+[score,target]].dropna().copy()
    if u.empty:return pd.Series(dtype=float)
    u['x']=u.groupby(keys)[score].rank();u['y']=u.groupby(keys)[target].rank();g=u.groupby(keys)
    u['x']-=g.x.transform('mean');u['y']-=g.y.transform('mean');u['xy']=u.x*u.y;u['xx']=u.x*u.x;u['yy']=u.y*u.y
    s=u.groupby(keys)[['xy','xx','yy']].sum();n=u.groupby(keys).size();r=s.xy/np.sqrt(s.xx*s.yy+EPS);r[n<minn]=np.nan;return r

def panel(raw):
    d=raw.sort_values(['code','date']).copy();g=d.groupby('code',sort=False)
    d['r1']=g.close.transform(lambda s:np.log(s).diff());d['sig20']=np.sqrt(g.r1.transform(lambda s:s.shift(1).pow(2).rolling(20,min_periods=12).mean()));d['u']=(d.r1/(d.sig20+EPS)).clip(-12,12)
    for h in [1,5,10,20,60]:d[f'mom{h}']=g.r1.transform(lambda s,h=h:s.rolling(h,min_periods=max(1,int(.7*h))).sum())
    for w in [5,20,60]:d[f'vol{w}']=np.sqrt(g.r1.transform(lambda s,w=w:s.pow(2).rolling(w,min_periods=max(3,int(.7*w))).mean()))
    rg=np.log(d.high/d.low);d['range1']=rg
    for w in [5,20]:d[f'range{w}']=rg.groupby(d.code).transform(lambda s,w=w:s.rolling(w,min_periods=max(3,int(.7*w))).mean())
    d['amount_log']=np.log1p(d.amount);d['amount_z']=prior_z(d.amount_log,d.code);d['volume_z']=prior_z(np.log1p(d.volume),d.code);d['turnover_z']=prior_z(d.turnover,d.code);d['range_z']=prior_z(rg,d.code)
    d['amihud_z']=prior_z(np.log1p(d.r1.abs()/(d.amount/1e8+EPS)),d.code);d['gap']=np.log(d.open/g.close.shift(1));d['close_loc']=(d.close-d.low)/(d.high-d.low+EPS)-.5
    d['skew20']=g.r1.transform(lambda s:s.rolling(20,min_periods=14).skew());d['kurt20']=g.r1.transform(lambda s:s.rolling(20,min_periods=14).kurt())
    pi={};zpi={}
    for s in [2,4,8,16]:
        m=g.u.transform(lambda x,s=s:x.rolling(s,min_periods=s).mean());q=g.u.transform(lambda x,s=s:x.pow(2).rolling(s,min_periods=s).mean());tau=(q-m*m).clip(lower=0);grad=(m-m.groupby(d.code).shift(s))/s;pi[s]=-tau*grad;zpi[s]=prior_z(pi[s],d.code)
    d['eps']=d.u.abs().pow(3)*(1+d.amount_z.clip(lower=0)+d.range_z.clip(lower=0));ages={}
    for w in [10,20,40]:ages[w]=g.eps.transform(lambda s,w=w:pd.Series(age_max(s.to_numpy(),w),index=s.index))
    cands=[]
    for a,b,c in [(2,4,8),(4,8,16)]:
        common=pd.concat([zpi[a],zpi[b],zpi[c]],axis=1).median(axis=1,skipna=False);coh=.5*(np.sign(pi[a])*np.sign(pi[b])+np.sign(pi[b])*np.sign(pi[c]));flux=common*coh
        for w in [10,20,40]:
            an=ages[w]/(w-1)
            for dec in [3.,5.,10.]:
                name=f'cascade_{a}_{b}_{c}_age{w}_decay{int(dec)}';d[name]=prior_z(flux*(an-np.exp(-ages[w]/dec)),d.code);cands.append(name)
    d['liq20']=g.amount.transform(lambda s:s.shift(1).rolling(20,min_periods=12).median());d['lp']=d.groupby(['date','market']).liq20.rank(pct=True);d['bucket']=np.where(d.lp>=2/3,'high',np.where(d.lp>=1/3,'secondary','tail'));d['cell']=d.market+'_'+d.bucket
    d['fwd5_raw']=np.log(g.close.shift(-5)/g.open.shift(-1));d['target5']=d.fwd5_raw-d.groupby(['date','market']).fwd5_raw.transform('median')
    base=['mom1','mom5','mom10','mom20','mom60','vol5','vol20','vol60','range5','range20','amount_z','volume_z','turnover_z','amihud_z','gap','close_loc','skew20','kurt20']
    for c in base:d['cs_'+c]=cs_z(d,c)
    return d,cands,['cs_'+c for c in base]

def pmask(d,p):
    return d.date.between(*( (DEV0,DEV1) if p=='development' else (VAL0,VAL1) if p=='validation' else (LAT0,LAT1) ))

def select(d,cands):
    q=d[pmask(d,'development')&d.bucket.isin(['high','secondary'])];rows=[]
    for c in cands:
        m=rankic(q,c,minn=60).mean();sgn=1 if m>=0 else -1;x=q[['date','cell','target5',c]].copy();x['f']=sgn*x[c];ic=rankic(x,'f',keys=['date','cell'],minn=12);cm=ic.groupby(level='cell').mean().to_dict();h1=rankic(x[x.date<='2020-12-31'],'f',minn=50).mean();h2=rankic(x[x.date>='2021-01-01'],'f',minn=50).mean();v=[cm.get(k,np.nan) for k in ['SH_high','SH_secondary','SZ_high','SZ_secondary']]+[h1,h2];v=[z for z in v if np.isfinite(z)];score=min(v)+.35*np.mean(v);rows.append({'candidate':c,'orientation':sgn,'score':score,'dev_mean':sgn*m,'half1':h1,'half2':h2,**cm})
    t=pd.DataFrame(rows).sort_values('score',ascending=False).reset_index(drop=True);return t.loc[0,'candidate'],float(t.loc[0,'orientation']),t

def direct(d):
    out=[]
    for p in ['development','validation','later']:
        q=d[pmask(d,p)&d.bucket.isin(['high','secondary'])]
        for s,x in [('all',q),('SH',q[q.market=='SH']),('SZ',q[q.market=='SZ']),('high',q[q.bucket=='high']),('secondary',q[q.bucket=='secondary'])]:
            z=rankic(x,'factor',minn=25 if s=='all' else 12);out.append({'period':p,'scope':s,'rankic':z.mean(),'positive_share':(z>0).mean(),'n_dates':z.notna().sum()})
    return pd.DataFrame(out)

def model_inc(d,base):
    cols=['date','target5','factor']+base;w=d[d.bucket.isin(['high','secondary'])][cols].replace([np.inf,-np.inf],np.nan).dropna();dev=w[pmask(w,'development')]
    if len(dev)>500000:dev=dev.sample(500000,random_state=SEED)
    early=dev[dev.date<'2022-01-01'];tune=dev[dev.date>='2022-01-01'];sb=StandardScaler().fit(early[base]);a=[]
    for al in [.1,1,10,100,1000]:
        m=Ridge(alpha=al).fit(sb.transform(early[base]),early.target5);a.append((mean_squared_error(tune.target5,m.predict(sb.transform(tune[base]))),al))
    al=min(a)[1];sp=StandardScaler().fit(dev[base+['factor']]);sb=StandardScaler().fit(dev[base]);mb=Ridge(alpha=al).fit(sb.transform(dev[base]),dev.target5);mp=Ridge(alpha=al).fit(sp.transform(dev[base+['factor']]),dev.target5);out=[]
    for p in ['validation','later']:
        x=w[pmask(w,p)].copy();pb=mb.predict(sb.transform(x[base]));pp=mp.predict(sp.transform(x[base+['factor']]));m0=mean_squared_error(x.target5,pb);m1=mean_squared_error(x.target5,pp);x['pb']=pb;x['pp']=pp;out.append({'period':p,'alpha':al,'n':len(x),'mse_base':m0,'mse_plus':m1,'mse_gain_pct':100*(m0-m1)/m0,'rankic_base':rankic(x,'pb').mean(),'rankic_plus':rankic(x,'pp').mean()})
    return pd.DataFrame(out),{'alpha':al,'tuning':a,'dev_rows':len(dev)}

def indep(d,base):
    cols=['date','target5','factor']+base;w=d[d.bucket.isin(['high','secondary'])][cols].replace([np.inf,-np.inf],np.nan).dropna();dev=w[pmask(w,'development')]
    if len(dev)>500000:dev=dev.sample(500000,random_state=SEED)
    sc=StandardScaler().fit(dev[base]);m=Ridge(alpha=100).fit(sc.transform(dev[base]),dev.factor);out=[]
    for p in ['validation','later']:
        x=w[pmask(w,p)].copy();fit=m.predict(sc.transform(x[base]));x['resid']=x.factor-fit;cors=[]
        for c in base:
            y=x[['date','factor',c]].rename(columns={c:'parent'});cors.append((c,rankic(y,'factor','parent',minn=25).mean()))
        mc=max(cors,key=lambda z:abs(z[1]));out.append({'period':p,'reconstruction_r2':r2_score(x.factor,fit),'residual_rankic':rankic(x,'resid').mean(),'max_parent':mc[0],'max_parent_corr':mc[1]})
    return pd.DataFrame(out)

def block_idx(n,b,r):
    o=[]
    while len(o)<n:
        s=int(r.integers(0,max(1,n-b+1)));o.extend(range(s,min(n,s+b)))
    return np.array(o[:n])

def boot(s,reps=2000):
    x=s.dropna().to_numpy();r=np.random.default_rng(SEED+len(x));z=np.array([x[block_idx(len(x),20,r)].mean() for _ in range(reps)]);return {'n':len(x),'mean':x.mean(),'ci_low':np.quantile(z,.025),'ci_high':np.quantile(z,.975),'p_nonpositive':(z<=0).mean()}

def maxt(d,cands,sel,chosen):
    q=d[pmask(d,'validation')&d.bucket.isin(['high','secondary'])];ors=dict(zip(sel.candidate,sel.orientation));ss=[]
    for c in cands:q['tmp']=ors[c]*q[c];z=rankic(q,'tmp',minn=50);z.name=c;ss.append(z)
    M=pd.concat(ss,axis=1).sort_index();mu=M.mean();obs=mu/(M.std(ddof=1)/np.sqrt(M.count())+EPS);A=(M-mu).fillna(0).to_numpy();r=np.random.default_rng(SEED+77);mx=[]
    for _ in range(1000):
        z=A[block_idx(len(A),20,r)];t=z.mean(0)/(z.std(0,ddof=1)/math.sqrt(len(z))+EPS);mx.append(np.nanmax(t))
    mx=np.array(mx);return {'selected':chosen,'candidate_count':len(cands),'selected_t':obs[chosen],'selected_mean':mu[chosen],'p':(mx>=obs[chosen]).mean(),'q90':np.quantile(mx,.9),'q95':np.quantile(mx,.95)}

def costs(d,bps=15):
    q=d[d.bucket.isin(['high','secondary'])][['date','market','factor','fwd5_raw']].dropna().copy();q['rank']=q.groupby(['date','market']).factor.rank(pct=True);q['leg']=np.where(q['rank']>=.8,1,np.where(q['rank']<=.2,-1,0));q=q[q.leg!=0];m=q.groupby(['date','market','leg']).fwd5_raw.mean().unstack('leg');sp=(m.get(1,np.nan)-m.get(-1,np.nan)).groupby(level='date').mean();net=sp-2*bps/10000;out=[]
    for p in ['development','validation','later']:
        if p=='development':x=net[(net.index>=DEV0)&(net.index<=DEV1)]
        elif p=='validation':x=net[(net.index>=VAL0)&(net.index<=VAL1)]
        else:x=net[(net.index>=LAT0)&(net.index<=LAT1)]
        gr=sp.loc[x.index];ann=x.mean()*252/5;vol=x.std()*math.sqrt(252/5);out.append({'period':p,'n_dates':x.notna().sum(),'one_way_bps':bps,'gross_ann':gr.mean()*252/5,'net_ann':ann,'net_sharpe':ann/vol if vol>0 else np.nan})
    return pd.DataFrame(out)

def main():
    raw,audit=load();d,cands,base=panel(raw);chosen,sgn,selection=select(d,cands);d['factor']=cs_z(d.assign(tmp=sgn*d[chosen]),'tmp')
    di=direct(d);mi,mm=model_inc(d,base);ind=indep(d,base);co=costs(d);non=[]
    dates={x:i for i,x in enumerate(sorted(d.date.unique()))};d['pos']=d.date.map(dates)
    for p in ['development','validation','later']:
        for o in range(5):
            z=rankic(d[pmask(d,p)&d.bucket.isin(['high','secondary'])&(d.pos%5==o)],'factor',minn=25);non.append({'period':p,'offset':o,'rankic':z.mean(),'n_dates':z.notna().sum()})
    non=pd.DataFrame(non);bs={p:boot(rankic(d[pmask(d,p)&d.bucket.isin(['high','secondary'])],'factor',minn=50)) for p in ['validation','later']};mt=maxt(d,cands,selection,chosen)
    def v(t,p,c,s=None):
        x=t[t.period==p];x=x[x.scope==s] if s is not None and 'scope' in x else x;return float(x.iloc[0][c])
    crit={'val_ic':v(di,'validation','rankic','all')>.015,'later_ic':v(di,'later','rankic','all')>.01,'val_markets':v(di,'validation','rankic','SH')>0 and v(di,'validation','rankic','SZ')>0,'later_markets':v(di,'later','rankic','SH')>0 and v(di,'later','rankic','SZ')>0,'val_mse':v(mi,'validation','mse_gain_pct')>0,'later_mse':v(mi,'later','mse_gain_pct')>0,'val_resid':v(ind,'validation','residual_rankic')>0,'later_resid':v(ind,'later','residual_rankic')>0,'val_cost':v(co,'validation','net_ann')>0,'later_cost':v(co,'later','net_ann')>0,'low_r2':v(ind,'validation','reconstruction_r2')<.5,'max_t':mt['p']<.1}
    decision='admit_public_daily_replication_component_default_off' if all(crit.values()) else 'reject_or_retain_research_only_no_trade_catalog_entry'
    summary={'status':'completed','data':audit,'periods':{'dev':'2019-2022','validation':'2023-2024','later':'2025-2026-11-15'},'candidate_count':len(cands),'selected':chosen,'orientation':sgn,'definition':'strict-prior coherent subgrid flux times dissipation-age switch','direct':di.to_dict('records'),'model':mi.to_dict('records'),'model_meta':mm,'independence':ind.to_dict('records'),'cost':co.to_dict('records'),'nonoverlap':non.to_dict('records'),'bootstrap':bs,'max_t':mt,'criteria':crit,'decision':decision,'limitations':['current-list survivorship bias','daily external replication, not local minute panel','public vendor adjusted history','cost bridge excludes execution frictions beyond fixed bps']}
    for name,obj in [('candidate_selection.csv',selection),('direct_rankic.csv',di),('ridge_increment.csv',mi),('independence.csv',ind),('cost_15bp.csv',co),('nonoverlap.csv',non)]:obj.to_csv(OUT/name,index=False)
    for name,obj in [('summary.json',summary),('data_audit.json',audit),('bootstrap.json',bs),('max_t.json',mt)]:(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    report=['# A-share coarse-grained cascade public daily validation','',f'Selected: `{chosen}`, orientation {sgn:+.0f}.','', 'This is an external daily-resolution replication; it does not replace the local minute-panel audit.','', '## Direct RankIC','',di.to_markdown(index=False),'','## Ridge increment','',mi.to_markdown(index=False),'','## Independence','',ind.to_markdown(index=False),'','## 15bp cost bridge','',co.to_markdown(index=False),'','## Max-T','',f'```json\n{json.dumps(mt,indent=2,default=float)}\n```','','## Decision','',f'`{decision}`']
    (OUT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8');print(json.dumps({'decision':decision,'selected':chosen,'criteria':crit},indent=2),flush=True)
if __name__=='__main__':main()
