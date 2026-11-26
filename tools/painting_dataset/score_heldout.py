"""Frozen final-turn heldout scoring; never runs a policy or optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tools.painting_dataset.perceptual_reward import SpatialPerceptual, MODEL, REVISION, LAYERS
from tools.painting_dataset.review_reward import load
from tools.painting_dataset.painting_reward import score


def summary(rows):
    valid = [r for r in rows if r['valid']]
    return {'n':len(rows),'mean':float(np.mean([r['reward'] for r in rows])), 'valid_conditional_mean':float(np.mean([r['reward'] for r in valid])) if valid else None,'valid_count':len(valid),'valid_rate':len(valid)/len(rows), 'token_budget_hits_all_turns':sum(r['token_budget_hits_all_turns'] for r in rows),'token_budget_hits_final':sum(r['token_budget_hits_final'] for r in rows),'truncations_all_turns':sum(r['truncations_all_turns'] for r in rows),'truncations_final':sum(r['truncations_final'] for r in rows)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('episodes',type=Path); p.add_argument('jobs',type=Path); p.add_argument('controls',type=Path); p.add_argument('output',type=Path); args=p.parse_args()
    jobs={j['seed']:j for j in json.loads(args.jobs.read_text())}; episodes=[]
    for path in sorted(args.episodes.glob('ep-*/episode.json')):
        meta=json.loads(path.read_text()); assert meta['complete'] and len(meta['turns'])==3
        episodes.append((path.parent,meta))
    assert len(episodes)==len(jobs) and {m['seed'] for _,m in episodes}==set(jobs)
    args.output.mkdir(parents=True,exist_ok=False); metric=SpatialPerceptual(); rows=[]
    audit=json.loads(args.controls.read_text()); controls={}
    for aid in sorted({j['artwork_id'] for j in jobs.values()}):
        target=next(j['target'] for j in jobs.values() if j['artwork_id']==aid)
        subset=[r for r in audit['controls'] if r['artwork_id']==aid]
        result=metric.compare([load(target)]+[load(r['image']) for r in subset])
        controls[aid]={r['control']:m for r,m in zip(subset,result[1:],strict=True)}
    for directory,meta in episodes:
        t=meta['turns'][-1]; j=jobs[meta['seed']]; valid=t['valid']; cap=meta['config']['max_tokens']
        row={'seed':meta['seed'],'artwork_id':j['artwork_id'],'episode':str(directory.resolve()),'target':j['target'],'valid':valid,'reward':0.,'token_budget_hits_all_turns':sum(x['tokens']>=cap for x in meta['turns']),'token_budget_hits_final':t['tokens']>=cap,'adapter_sha256':meta.get('adapter_sha256'),'error':t.get('error')}
        assert hashlib.sha256((directory/'target.png').read_bytes()).hexdigest()==meta['target_sha256']
        with Image.open(j['target']) as original, Image.open(directory/'target.png') as copied:
            assert np.array_equal(np.asarray(original.convert('RGB')),np.asarray(copied.convert('RGB')))
        finishes=[json.loads((directory/('turn-'+str(x['turn']))/'response.json').read_text())['finish_reason'] for x in meta['turns']]
        row.update(finish_reasons=finishes,truncations_all_turns=finishes.count('length'),truncations_final=finishes[-1]=='length')
        if valid:
            render=json.loads((directory/'turn-2/render.json').read_text()); assert render['ok']
            row['image']=render['image']; row['perceptual']=metric.compare([load(directory/'target.png'),load(row['image'])])[1]
            row['reward']=float(np.clip(row['perceptual']['spatial_cosine'],0,1)); row['pixel']=score(directory/'target.png',row['image'])
        rows.append(row)
    rows.sort(key=lambda r:r['seed']); groups={a:summary([r for r in rows if r['artwork_id']==a]) for a in controls}
    report={'model':MODEL,'revision':REVISION,'layers':LAYERS,'jobs_sha256':hashlib.sha256(args.jobs.read_bytes()).hexdigest(),'metric_source_sha256':hashlib.sha256(Path(__file__).with_name('perceptual_reward.py').read_bytes()).hexdigest(),'reward':'clip unchanged aligned spatial cosine to[0,1]; final turn only; invalid=0','truncation_note':'Exact finish_reason=length from response.json; token budget hits separately reported.','summary':summary(rows),'per_artwork':groups,'rows':rows,'controls':controls}
    (args.output/'scores.json').write_text(json.dumps(report,indent=2)+'\n')
    canvas=Image.new('RGB',(900,220*len(controls)),'white');draw=ImageDraw.Draw(canvas)
    for i,aid in enumerate(controls):
        group=[r for r in rows if r['artwork_id']==aid]; items=[('target '+aid,load(group[0]['target']))]+[(str(r['seed'])+' '+format(r['reward'],'.3f'),load(r['image']) if r['valid'] else Image.new('RGB',(160,160),'black')) for r in group]
        for col,(label,im) in enumerate(items):
            im=ImageOps.contain(im,(174,174));canvas.paste(im,(col*180+(180-im.width)//2,i*220));draw.text((col*180+2,i*220+178),label,fill='black')
    canvas.save(args.output/'final-candidates.jpg',quality=92)
    print(json.dumps({'summary':report['summary'],'per_artwork':groups},indent=2))

if __name__=='__main__': main()
