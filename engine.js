/* Hand-written forward pass. Mirrors arcstate/model.py line for line.
   Verified against PyTorch at load time via the parity fixture. */
const EPS=1e-5;
function mmT(x,rows,din,W,dout){ // x:(rows,din) W:(dout,din) -> (rows,dout)
  const o=new Float32Array(rows*dout);
  for(let r=0;r<rows;r++){const xo=r*din,oo=r*dout;
    for(let j=0;j<dout;j++){let s=0,wo=j*din;
      for(let i=0;i<din;i++) s+=x[xo+i]*W[wo+i];
      o[oo+j]=s;}}
  return o;}
function relu_(a){for(let i=0;i<a.length;i++) if(a[i]<0)a[i]=0; return a;}
function add_(a,b){for(let i=0;i<a.length;i++)a[i]+=b[i]; return a;}

class Engine{
  constructor(spec){
    this.c=spec.config; this.meta=spec.meta;
    const raw=atob(spec.weights_b64);
    const bytes=new Uint8Array(raw.length);
    for(let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
    this.buf=new Float32Array(bytes.buffer);
    this.parity=spec.parity;
  }
  T(n){const m=this.meta[n]; return this.buf.subarray(m.off,m.off+m.n);}
  lowrank(p,v,rows){ // relu-lowrank block, BDH Sec 5.2
    const {d,n}=this.c;
    const x=relu_(mmT(v,rows,d,this.T(p+".up.weight"),n));
    const e=mmT(x,rows,n,this.T(p+".e.weight"),d);
    const y=relu_(mmT(e,rows,d,this.T(p+".dd.weight"),n));
    const s=new Float32Array(x.length);
    for(let i=0;i<x.length;i++) s[i]=x[i]+y[i];
    return mmT(s,rows,n,this.T(p+".dn.weight"),d);
  }
  sharp(p,h,rows,dout){ // k = relu(Wh+b)^2
    const k=mmT(h,rows,this.c.d,this.T(p+".w.weight"),dout), b=this.T(p+".b");
    for(let r=0;r<rows;r++){const o=r*dout;
      for(let j=0;j<dout;j++){let v=k[o+j]+b[j]; k[o+j]=v>0?v*v:0;}}
    return k;
  }
  memEmbed(tok){ // (T,5) -> (T,d) factored
    const {d,W}=this.c, T=tok.length, o=new Float32Array(T*d);
    const cin=this.T("task_writer.emb.cin.weight"),cout=this.T("task_writer.emb.cout.weight"),
          pos=this.T("task_writer.emb.pos.weight"),dem=this.T("task_writer.emb.demo.weight");
    for(let t=0;t<T;t++){const b=t*d,r=tok[t];
      for(let i=0;i<24;i++) o[b+i]=cin[r[0]*24+i];
      for(let i=0;i<24;i++) o[b+24+i]=cout[r[1]*24+i];
      const p=r[2]*W+r[3];
      for(let i=0;i<12;i++) o[b+48+i]=pos[p*12+i];
      for(let i=0;i<4;i++)  o[b+60+i]=dem[r[4]*4+i];}
    return o;
  }
  qEmbed(tok,pfx){ // (L,5) -> (L,d), colour in the SAME block memory keys on
    const {d,W}=this.c,T=tok.length,o=new Float32Array(T*d);
    const col=this.T(pfx+"col.weight"),pos=this.T(pfx+"pos.weight");
    for(let t=0;t<T;t++){const b=t*d,r=tok[t];
      for(let i=0;i<24;i++) o[b+i]=col[r[0]*24+i];
      const p=r[2]*W+r[3];
      for(let i=0;i<12;i++) o[b+48+i]=pos[p*12+i];}
    return o;
  }
  write(emb,rows,pfx){ // additive outer-product state:  S = sum_t k_t (x) v_t
    const {d,dk,dv}=this.c;
    const h=emb.slice(); add_(h,this.lowrank(pfx+".blk",emb,rows));
    const k=this.sharp(pfx+".wk",h,rows,dk);
    const v=mmT(h,rows,d,this.T(pfx+".wv.weight"),dv);
    const S=new Float32Array(dk*dv), z=new Float32Array(dk);
    for(let t=0;t<rows;t++){const ko=t*dk,vo=t*dv;
      for(let a=0;a<dk;a++){const ka=k[ko+a]; if(ka===0)continue; z[a]+=ka;
        const so=a*dv; for(let b=0;b<dv;b++) S[so+b]+=ka*v[vo+b];}}
    return {S,z,k,v};
  }
  read(S,z,q,rows){ // (q.S)/(q.z)
    const {dk,dv}=this.c,o=new Float32Array(rows*dv);
    for(let r=0;r<rows;r++){const qo=r*dk; let den=0;
      for(let a=0;a<dk;a++) den+=q[qo+a]*z[a];
      den+=EPS; const oo=r*dv;
      for(let a=0;a<dk;a++){const qa=q[qo+a]; if(qa===0)continue; const so=a*dv;
        for(let b=0;b<dv;b++) o[oo+b]+=qa*S[so+b];}
      for(let b=0;b<dv;b++) o[oo+b]/=den;}
    return o;
  }
  solve(qtok,S,z,R){
    const {d,dk,dv,heads:hd,NC}=this.c, L=qtok.length;
    let Hs=this.qEmbed(qtok,"q_emb.");
    const qm=this.write(this.qEmbed(qtok,"query_writer.emb."),L,"query_writer");
    const traj=[];
    for(let r=0;r<R;r++){
      const qs=this.sharp("reason.qs",Hs,L,dk*hd);
      const qq=this.sharp("reason.qq",Hs,L,dk*hd);
      const a=this.read(S,z,qs,L*hd), b=this.read(qm.S,qm.z,qq,L*hd);
      const u=Hs.slice();
      add_(u,mmT(a,L,dv*hd,this.T("reason.os.weight"),d));
      add_(u,mmT(b,L,dv*hd,this.T("reason.oq.weight"),d));
      const f=this.lowrank("reason.blk",u,L); add_(f,u);
      // LayerNorm
      const g=this.T("reason.norm.weight"),bb=this.T("reason.norm.bias");
      for(let i=0;i<L;i++){const o=i*d; let m=0;
        for(let j=0;j<d;j++)m+=f[o+j]; m/=d; let v=0;
        for(let j=0;j<d;j++){const t=f[o+j]-m; v+=t*t;} v=Math.sqrt(v/d+1e-5);
        for(let j=0;j<d;j++) f[o+j]=((f[o+j]-m)/v)*g[j]+bb[j];}
      Hs=f;
      traj.push(mmT(Hs,L,d,this.T("out.weight"),NC));
    }
    return {logits:traj[traj.length-1],traj};
  }
  buildState(memtok){
    const {dk,dv}=this.c;
    if(!memtok||memtok.length===0)
      return {S:new Float32Array(dk*dv),z:new Float32Array(dk),k:null,v:null};
    return this.write(this.memEmbed(memtok),memtok.length,"task_writer");
  }
  argmax(lg,L){const {NC}=this.c,o=new Int32Array(L);
    for(let i=0;i<L;i++){let bi=0,bv=-1e30;
      for(let j=0;j<NC;j++){const v=lg[i*NC+j]; if(v>bv){bv=v;bi=j;}} o[i]=bi;}
    return o;}
  verify(){ // parity against PyTorch
    const p=this.parity, st=this.buildState(p.mem);
    const {logits}=this.solve(p.query,st.S,st.z,3);
    const am=Array.from(this.argmax(logits,p.query.length));
    const cellsOk=am.every((v,i)=>v===p.argmax[i]);
    let maxAbs=0;
    for(let i=0;i<p.logits_sample.length;i++)
      maxAbs=Math.max(maxAbs,Math.abs(logits[i]-p.logits_sample[i]));
    return {ok:cellsOk&&maxAbs<2e-2, maxAbs, cellsOk};
  }
}
