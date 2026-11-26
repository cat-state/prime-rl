"""Paired-seed comparison of frozen heldout reward reports."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('updated',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    base=json.loads(a.base.read_text());new=json.loads(a.updated.read_text())
    for key in ('model','revision','layers','jobs_sha256','metric_source_sha256','reward'):
        assert base[key]==new[key],key
    old={r['seed']:r for r in base['rows']}; changed={r['seed']:r for r in new['rows']};assert old.keys()==changed.keys()
    pairs=[]
    for seed,b in old.items():
        n=changed[seed];assert b['artwork_id']==n['artwork_id']
        pairs.append({'seed':seed,'artwork_id':b['artwork_id'],'base_reward':b['reward'],'updated_reward':n['reward'],'delta':n['reward']-b['reward'],'base_valid':b['valid'],'updated_valid':n['valid'],'both_valid':b['valid'] and n['valid']})
    def aggregate(rows):
        both=[r['delta'] for r in rows if r['both_valid']]
        return {'n':len(rows),'paired_mean_delta':float(np.mean([r['delta'] for r in rows])),'both_valid_count':len(both),'both_valid_mean_delta':float(np.mean(both)) if both else None,'improved':sum(r['delta']>1e-5 for r in rows),'declined':sum(r['delta']< -1e-5 for r in rows),'invalid_to_valid':sum(not r['base_valid'] and r['updated_valid'] for r in rows),'valid_to_invalid':sum(r['base_valid'] and not r['updated_valid'] for r in rows)}
    report={'base':base['summary'],'updated':new['summary'],'paired':aggregate(pairs),'per_artwork':{aid:aggregate([r for r in pairs if r['artwork_id']==aid]) for aid in sorted({r['artwork_id'] for r in pairs})},'pairs':pairs,'interpretation_limit':'Four artworks and four seeds each; artwork-disjoint, not artist-disjoint. Conditional means select different valid populations. No statistical generalization or visual-quality claim from reward alone.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
