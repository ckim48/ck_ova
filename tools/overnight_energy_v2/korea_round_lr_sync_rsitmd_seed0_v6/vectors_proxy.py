def vectors(engine,theta,plan,queue,t,fl,kind,ids,proxy,teachers,crossmodal):
    d=len(theta)-1;l=np.zeros((10,d));u=np.zeros_like(l);stats=[]
    pools=retained_pools(plan,queue,t)
    if kind in ['fl','joint'] and fl<FL_STEPS:
        k=21+fl;lr=core.lr_at(k);total=sum(map(len,pools))
        for user,pool in enumerate(pools):
            batch=np.random.default_rng(730000+k*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
            g,_=engine.gradient(theta,batch,k)
            l[user]=(-lr*len(pool)/total*g[:-1]).cpu().numpy()
    total=sum(len(queue.items[i]['images']) for i in ids)
    for rid in ids:
        req=queue.items[rid];user=req['user'];step=req['steps'];fids=req['images']
        if rid not in teachers:teachers[rid]=core.Teacher(engine,theta)
        cids=[c for i in fids for c in engine.data.caption_ids[i]]
        fg,loss=local_gradient(engine,theta,fids,cids,proxy,crossmodal)
        seed=910000+req['burst']*1000+user*10+step
        rids=np.random.default_rng(seed).choice(pools[user],min(16,len(pools[user])),replace=False).astype(int).tolist()
        rcids=engine.data.training_caption_ids(rids,21+step)
        rg,reg=local_gradient(engine,theta,rids,rcids,teachers[rid],crossmodal)
        u[user]+=(-core.lr_at(21+fl)*len(fids)/total*(fg+rg)[:-1]).cpu().numpy()
        stats.append(dict(request=rid,step=step+1,learning_rate_index=21+fl,FL_schedule_learning_rate=core.lr_at(21+fl),FU_learning_rate=core.lr_at(21+fl),forget_loss=loss,preservation_loss=reg,
                          forget_gradient_norm=float(fg.norm()),preserve_gradient_norm=float(rg.norm()),
                          images=len(fids),retained_images=rids))
    return l,u,stats
