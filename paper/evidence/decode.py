from pathlib import Path
import re,base64,struct,numpy as np
p=Path('work/paper-figures');s=(p/'console.log').read_text('utf8',errors='replace')
for name in ['flow','elastic']:
 m=re.search('PAPER_DATA_'+name+r' ([A-Za-z0-9+/=]+)',s)
 if not m:raise ValueError(name+' missing')
 b=base64.b64decode(m[1]);(p/(name+'.bin')).write_bytes(b);ne,np_,nr,nc=struct.unpack('>4i',b[:16]);a=np.frombuffer(b,dtype='>f8',offset=16,count=ne*np_).reshape(ne,np_).astype(float);t=np.frombuffer(b,dtype='>i4',offset=16+8*ne*np_).reshape(nr,nc).T.astype(np.int64)
 np.savez(p/(name+'_solution.npz'),values=a,tetrahedra=t)
 print(name,a.shape,t.shape, 'ranges',[(float(v.min()),float(v.max())) for v in a], 'connectivity',t.min(),t.max())
