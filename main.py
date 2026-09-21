import argparse
import json
import time
from pathlib import Path
import numpy as np
from common import BASE, finish, save_json, plt
from model import generate, solve, metrics, canonical


def main():
    p=argparse.ArgumentParser(description='Вариант 16: раскраска графа')
    p.add_argument('--config',type=Path,default=BASE/'config.json');p.add_argument('--runs',type=int);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path,default=BASE/'results')
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if a.runs is not None:c['runs']=a.runs
    if a.seed is not None:c['first_seed']=a.seed
    if c['runs']<1 or c['population']<2 or c['generations']<1 or c['first_seed']<0 or c['penalty']<=30 or not all(0<=c[k]<=1 for k in ['mutation_probability','crossover_probability']):p.error('Некорректная конфигурация (штраф должен быть >30)')
    graph=json.loads((BASE/'data/graph.json').read_text(encoding='utf-8'))
    rows=[];traces={};solutions=[]
    for method,handling,mutation,baseline in [('Penalty-reset','penalty','reset',False),('Repair-reset','repair','reset',False),('Repair-copy','repair','copy',False),('Random-greedy','repair','reset',True)]:
        traces[method]=[]
        for i in range(c['runs']):
            seed=c['first_seed']+i;start=time.perf_counter()
            result,best,trace=solve(c,graph,seed,handling,mutation,baseline)
            rows.append(dict(method=method,seed=seed,**result,seconds=time.perf_counter()-start))
            traces[method].append(trace);solutions.append(dict(method=method,seed=seed,assignment=best.tolist(),**result))
        print(method,'completed',flush=True)
    finish(a.output,c,rows,traces,'Best penalized objective (minimize)')
    save_json(a.output/'solutions.json',solutions)
    best=min((s for s in solutions if s['conflicts']==0),key=lambda x:x['score'])
    angles=np.linspace(0,2*np.pi,graph['vertices'],endpoint=False);xy=np.c_[np.cos(angles),np.sin(angles)]
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    for u,v in graph['edges']:ax.plot(xy[[u,v],0],xy[[u,v],1],color='#cbd5e1',lw=.7,zorder=0)
    ax.scatter(*xy.T,c=best['assignment'],cmap='tab10',s=430,edgecolors='white',linewidths=2)
    for i,(x,y) in enumerate(xy):ax.text(x,y,str(i),ha='center',va='center',fontsize=9)
    ax.set_title(f"Best feasible coloring: {best['colors']} colors");ax.axis('off');fig.savefig(a.output/'coloring.png',dpi=150);plt.close(fig)
    valid1=np.array(graph['known_coloring']);valid2=np.arange(graph['vertices'])
    invalid1=np.zeros(graph['vertices'],dtype=int);invalid2=valid1.copy();invalid2[1]=invalid2[0]
    save_json(a.output/'examples.json',[dict(label=label,vector=x.tolist(),colors=metrics(x,graph)[0],conflicts=metrics(x,graph)[1]) for label,x in [('valid_5_colors',valid1),('valid_30_colors',valid2),('invalid_all_same',invalid1),('invalid_clique_conflict',invalid2)]])

if __name__=='__main__':main()
