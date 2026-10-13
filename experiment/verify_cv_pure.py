import csv, math, random
rows=list(csv.DictReader(open('results/bitstream_features_controlled.csv')))
EXCLUDE={'video_file','label','codec','q1','q2','delta_qp','sample_path','src_path','group_id','error','crf','level','height','width','duration_s','n_frames_decoded','codec_name','pixel_format','profile'}
feat=[k for k in rows[0].keys() if k not in EXCLUDE]
nfeat=len(feat)
X=[]; y=[]; src=[]
for r in rows:
    X.append([float(r[k]) for k in feat]); y.append(1 if r['label']=='double' else 0); src.append(r['video_file'])
N=len(X)
# standardize
mu=[sum(X[i][j] for i in range(N))/N for j in range(nfeat)]
sd=[math.sqrt(sum((X[i][j]-mu[j])**2 for i in range(N))/N) for j in range(nfeat)]
sd=[1.0 if s==0 else s for s in sd]
Xs=[[(X[i][j]-mu[j])/sd[j] for j in range(nfeat)] for i in range(N)]

def matvec(M,v):
    return [sum(M[i][k]*v[k] for k in range(len(v))) for i in range(len(M))]
def dot(a,b): return sum(a[i]*b[i] for i in range(len(a)))
def sigmoid(z): return 1.0/(1.0+math.exp(-z))

def lr_train(Xtr,ytr):
    # Xtr list of rows; add bias
    m=len(Xtr); d=nfeat+1
    X1=[[1.0]+r for r in Xtr]
    w=[0.0]*d
    for _ in range(80):
        # gradient and Hessian
        g=[0.0]*d; H=[[0.0]*d for _ in range(d)]
        for i in range(m):
            z=dot(X1[i],w); p=sigmoid(z); e=p-ytr[i]
            for a in range(d):
                g[a]+=X1[i][a]*e
                for b in range(d):
                    H[a][b]+=X1[i][a]*X1[i][b]*p*(1-p)
        for a in range(d): H[a][a]+=1e-6
        # solve H dw = -g (Gaussian elimination)
        A=[row[:]+[-g[r]] for r,row in enumerate(H)]
        for c in range(d):
            piv=max(range(c,d), key=lambda r: abs(A[r][c]))
            A[c],A[piv]=A[piv],A[c]
            pv=A[c][c]
            for j in range(d+1): A[c][j]/=pv
            for r in range(d):
                if r!=c:
                    f=A[r][c]
                    for j in range(d+1): A[r][j]-=f*A[c][j]
        w=[A[r][d] for r in range(d)]
    return w

def auc(yy,sc):
    o=sorted(range(len(yy)), key=lambda i: sc[i]); n=len(yy)
    rank=[0]*n
    # handle ties: assign average rank
    i=0
    while i<n:
        j=i
        while j+1<n and sc[o[j+1]]==sc[o[i]]: j+=1
        avg=(i+j)/2.0+1.0
        for k in range(i,j+1): rank[o[k]]=avg
        i=j+1
    n1=sum(yy); n0=n-n1
    if n1==0 or n0==0: return float('nan')
    return (sum(rank[i] for i in range(n) if yy[i]==1)-n1*(n1+1)/2.0)/(n1*n0)

def sgkf(src,n_splits,seed):
    rng=random.Random(seed)
    groups=list(set(src)); gorder=rng.sample(groups,len(groups))
    gidx={g:i for i,g in enumerate(groups)}; fold_of=[0]*len(src)
    for gi,g in enumerate(gorder):
        f=gi%n_splits
        for i in range(len(src)):
            if src[i]==g: fold_of[i]=f
    tr=[[i for i in range(N) if fold_of[i]!=f] for f in range(n_splits)]
    te=[[i for i in range(N) if fold_of[i]==f] for f in range(n_splits)]
    return tr,te

def skf(y,n_splits,seed):
    rng=random.Random(seed); folds=[[] for _ in range(n_splits)]
    for lab in [0,1]:
        idx=[i for i in range(N) if y[i]==lab]; rng.shuffle(idx)
        for i,ix in enumerate(idx): folds[i%n_splits].append(ix)
    tr=[[i for f in range(n_splits) if f!=k for i in folds[f]] for k in range(n_splits)]
    te=[list(folds[k]) for k in range(n_splits)]
    return tr,te

def run(sa):
    aucs=[]
    for seed in range(10):
        tr,te=sgkf(src,5,seed) if sa else skf(y,5,seed)
        for a,b in zip(tr,te):
            if len(set(y[i] for i in a))<2 or len(set(y[i] for i in b))<2: continue
            w=lr_train([Xs[i] for i in a],[y[i] for i in a])
            sc=[sigmoid(dot([1.0]+Xs[i],w)) for i in b]
            aucs.append(auc([y[i] for i in b],sc))
    mean=sum(aucs)/len(aucs)
    var=sum((a-mean)**2 for a in aucs)/len(aucs)
    return mean, math.sqrt(var)/math.sqrt(len(aucs)), len(aucs)

m,ci,n=run(True); print(f"SOURCE-AWARE LR  AUC={m:.4f} +/- {ci:.4f} (N={n})")
m,ci,n=run(False); print(f"RANDOM  LR      AUC={m:.4f} +/- {ci:.4f} (N={n})")
