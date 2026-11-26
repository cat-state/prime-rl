"""Freeze an explicit, hashed reward map from completed unchanged-metric scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('scores',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    report=json.loads(a.scores.read_text());rows=report['rows'];assert len({r['episode'] for r in rows})==len(rows)
    rewards={r['episode']:r['reward'] for r in rows}
    assert all(np.isfinite(r) and 0<=r<=1 for r in rewards.values())
    assert all(r['valid'] or r['reward']==0 for r in rows)
    groups={}
    for aid in report['per_artwork']:
        group=[r for r in rows if r['artwork_id']==aid];values=[r['reward'] for r in group];valid=[r['reward'] for r in group if r['valid']];c=report['controls'][aid]
        baseline=max(c['mean_color']['spatial_cosine'],c['fitted_gradient']['spatial_cosine']);noise=c['shuffled_pixels']['spatial_cosine']
        groups[aid]={'rewards':values,'std':float(np.std(values)),'valid_std':float(np.std(valid)) if valid else None,'range':max(values)-min(values),'valid_count':len(valid),'valid_above_wash':sum(v>baseline for v in valid),'valid_above_noise':sum(v>noise for v in valid),'wash_baseline':baseline,'noise_baseline':noise}
    with a.output.open('x') as f:json.dump(rewards,f,indent=2);f.write('\n')
    a.output.chmod(0o444)
    provenance={'status':'Frozen evaluation-derived map; preparation does not authorize optimizer execution','reward_map_sha256':sha(a.output),'scores_sha256':sha(a.scores),'metric_source_sha256':report['metric_source_sha256'],'model':report['model'],'revision':report['revision'],'layers':report['layers'],'per_artwork':groups,'episodes':[{'episode':r['episode'],'episode_json_sha256':sha(Path(r['episode'])/'episode.json'),'final_image_sha256':sha(r['image']) if r['valid'] else None} for r in rows]}
    with a.output.with_suffix('.provenance.json').open('x') as f:json.dump(provenance,f,indent=2);f.write('\n')
    print(json.dumps({'reward_map_sha256':provenance['reward_map_sha256'],'per_artwork':groups},indent=2))

if __name__=='__main__':main()
