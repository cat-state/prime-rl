import json,hashlib
from pathlib import Path
root=Path('/root/prime-rl/loop-validation');m=root/'painting-data-nga-v1/manifest.json';rows={r['id']:r for r in json.loads(m.read_text())['records']}
selected=['nga-106382','nga-126146','nga-69390','nga-154844','nga-270','nga-61375','nga-52201','nga-56291']
reason=['self portrait','floral still life','pastoral river with animals','rocky coast','Venetian architecture','domestic cradle scene','animal head','tavern interior']
old=json.loads((root/'agentic-pilot-jobs-v1.json').read_text());held=json.loads((root/'agentic-heldout-jobs-v1.json').read_text());excluded={j['artwork_id'] for j in old+held};assert not set(selected)&excluded
instruction=old[0]['instruction']+' WGSL syntax: let bindings and function parameters are immutable; use var for values you reassign (for example var col: vec3f = vec3f(0.0); col = mix(col, vec3f(1.0), 0.5);). Use f32 for scalar types. Match clamp bounds to the value type: clamp(col, vec3f(0.0), vec3f(1.0)); scalar clamp(x, 0.0, 1.0).'
jobs=[];provenance=[]
for i,aid in enumerate(selected):
 r=rows[aid]; assert r['split']=='train'; assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()==r['image_sha256']
 for j in range(4): jobs.append({'target':r['image'],'seed':3000+i*100+j*10,'artwork_id':aid,'instruction':instruction})
 provenance.append({**{k:r[k] for k in ['id','artist','title','split','artwork_key','image_sha256']},'selection_category':reason[i]})
assert len({r['artist'] for r in provenance})==8
path=root/'agentic-protocol-v2-train-jobs-v1.json';assert not path.exists();path.write_text(json.dumps(jobs,indent=2)+'\n')
control=[jobs[i*4] for i in range(4)];cp=root/'agentic-protocol-v2-control-jobs-v1.json';cp.write_text(json.dumps(control,indent=2)+'\n')
(root/'agentic-protocol-v2-train-provenance-v1.json').write_text(json.dumps({'protocol':'bounded-v2','turns':3,'seeds':32,'jobs_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'control_jobs_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'selection':'Predeclared metadata-only categories; no reward/model outputs consulted; four control jobs first seed of first four artworks. Same guidance for full-v1 versus bounded-v2 history control.','excluded_artwork_ids':sorted(excluded),'artworks':provenance},indent=2)+'\n')
print(path);print(cp)
