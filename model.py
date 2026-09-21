"""Раскраска: целочисленная хромосома, штраф и детерминированный ремонт."""
import numpy as np
from common import tournament


def generate(seed=2650717, n=30):
    rng = np.random.default_rng(seed)
    # Первые пять вершин образуют K5. Остальные рёбра только между классами.
    # Поэтому известен строгий оптимум chi(G)=5, но раскраска не подаётся в ГА.
    classes = np.arange(n) % 5
    edges = []
    for i in range(n):
        for j in range(i+1,n):
            if classes[i] != classes[j] and (j < 5 or rng.random() < .28):
                edges.append([i,j])
    return dict(seed=seed,vertices=n,edges=edges,clique_lower_bound=list(range(5)),known_coloring=classes.tolist())


def canonical(x):
    mapping = {}
    out = []
    for value in x:
        value = int(value)
        if value not in mapping: mapping[value] = len(mapping)
        out.append(mapping[value])
    return np.array(out,dtype=int)


def metrics(x, graph):
    edges = np.array(graph['edges'],dtype=int)
    conflicts = int(np.sum(x[edges[:,0]] == x[edges[:,1]])) if len(edges) else 0
    return len(set(map(int,x))), conflicts


def repair(x, graph):
    """Жадная перекраска конфликтующих вершин, при необходимости новый цвет."""
    x = canonical(x)
    neighbors = [set() for _ in x]
    for a,b in graph['edges']: neighbors[a].add(b); neighbors[b].add(a)
    for i in sorted(range(len(x)),key=lambda j:(-len(neighbors[j]),j)):
        used = {int(x[j]) for j in neighbors[i]}
        if x[i] in used:
            color = 0
            while color in used: color += 1
            x[i] = color
    return canonical(x)


def mutate(x, rng, probability, mode):
    x = x.copy()
    for i in np.flatnonzero(rng.random(len(x)) < probability):
        if mode == 'reset':
            x[i] = rng.integers(max(2,int(x.max())+2))
        else:
            # Перенос вершины в уже существующий класс; число цветов не растёт.
            x[i] = x[rng.integers(len(x))]
    return canonical(x)


def constructive(graph, rng):
    n = graph['vertices']; x = np.full(n,-1,dtype=int)
    neighbors=[set() for _ in range(n)]
    for a,b in graph['edges']: neighbors[a].add(b);neighbors[b].add(a)
    for i in rng.permutation(n):
        used={x[j] for j in neighbors[i]}; color=0
        while color in used: color+=1
        x[i]=color
    return canonical(x)


def solve(c,graph,seed,handling='repair',mutation='reset',baseline=False):
    rng=np.random.default_rng(seed);n=c['population'];d=graph['vertices']
    calls=0; feasible_count=0; feasible_best=None; feasible_colors=d+1
    def evaluate(x):
        nonlocal calls,feasible_count,feasible_best,feasible_colors
        colors,conflicts=metrics(x,graph);calls+=1
        if conflicts==0:
            feasible_count+=1
            if colors<feasible_colors: feasible_colors=colors;feasible_best=x.copy()
        return colors+c['penalty']*conflicts
    pop=np.array([canonical(rng.integers(0,7,d)) for _ in range(n)])
    if handling=='repair': pop=np.array([repair(x,graph) for x in pop])
    if baseline: pop=np.array([constructive(graph,rng) for _ in range(n)])
    scores=np.array([evaluate(x) for x in pop]); trace=[float(scores.min())]
    best=pop[np.argmin(scores)].copy();best_score=float(scores.min())
    for _ in range(c['generations']):
        children=[]
        if baseline:
            children=[constructive(graph,rng) for _ in range(n)]
        else:
            ia=tournament(rng,scores,n);ib=tournament(rng,scores,n)
            for a,b in zip(ia,ib):
                child=pop[a].copy()
                if rng.random()<c['crossover_probability']:
                    mask=rng.random(d)<.5;child[mask]=pop[b][mask]
                child=mutate(child,rng,c['mutation_probability'],mutation)
                children.append(repair(child,graph) if handling=='repair' else child)
        pop=np.array(children);scores=np.array([evaluate(x) for x in pop])
        idx=int(np.argmin(scores))
        if scores[idx]<best_score:best_score=float(scores[idx]);best=pop[idx].copy()
        if not baseline:
            idx=int(np.argmax(scores));pop[idx]=best;scores[idx]=best_score
        trace.append(best_score)
    return dict(score=best_score,colors=metrics(best,graph)[0],conflicts=metrics(best,graph)[1],evaluations=calls,feasible_fraction=feasible_count/calls,best_feasible_colors=feasible_colors if feasible_best is not None else None), best,trace
