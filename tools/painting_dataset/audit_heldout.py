"""CPU provenance/near-duplicate screening and fixed controls for frozen jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.painting_dataset.review_reward import load, variants
from tools.painting_dataset.painting_reward import score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    # Orthonormal factors do not affect the common non-DC coefficient median.
    basis = np.cos(np.pi * (np.arange(32)[None, :] + .5) * np.arange(8)[:, None] / 32)
    dct = basis @ a @ basis.T
    values = dct.ravel()[1:]
    return values > np.median(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path); p.add_argument('jobs', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = [r for r in json.loads(args.manifest.read_text())['records'] if r.get('image')]
    jobs = json.loads(args.jobs.read_text()); selected = {j['artwork_id'] for j in jobs}
    ids = {r['id']: r for r in rows}
    exact = {}
    for field in ('id', 'artwork_key', 'image_sha256'):
        groups = {}
        for r in rows: groups.setdefault(r[field], []).append(r['id'])
        exact[field] = [v for v in groups.values() if len(v) > 1]
    groups = {}
    for r in rows: groups.setdefault(r['published_image']['uuid'], []).append(r['id'])
    exact['published_image_uuid'] = [v for v in groups.values() if len(v)>1]
    hash_mismatches = [r['id'] for r in rows if digest(r['image']) != r['image_sha256']]
    hashes = [phash(r['image']) for r in rows]; distances = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]; distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            distances.append({'a':a['id'], 'b':b['id'], 'distance':distance, 'cross_split':a['split']!=b['split']})
    distances.sort(key=lambda x:x['distance'])
    controls = []
    for aid in sorted(selected):
        r = ids[aid]; assert r['split'] == 'heldout'
        assert all(j['target'] == r['image'] for j in jobs if j['artwork_id']==aid)
        target = load(r['image']); seed = int(hashlib.sha256(aid.encode()).hexdigest()[:8], 16)
        candidates = variants(target, target, seed)
        folder = args.output / aid; folder.mkdir()
        for name in ('mean_color','fitted_gradient','shuffled_pixels'):
            path = folder / (name+'.png'); candidates[name].save(path)
            controls.append({'artwork_id':aid,'control':name,'seed':seed,'image':str(path.resolve()),'sha256':digest(path),'pixel_diagnostic':score(r['image'],path)})
    report = {'manifest_sha256':digest(args.manifest),'jobs_sha256':digest(args.jobs),'downloaded_count':len(rows),'exact_duplicate_groups':exact,'hash_mismatches':hash_mismatches,'hash_method':'63-bit low-frequency DCT median hash, DC excluded; screening, not proof of absence','threshold':8,'near_duplicate_candidates':[d for d in distances if d['distance']<=8],'closest_cross_split_pairs':[d for d in distances if d['cross_split']][:20],'selected':[{k:ids[aid][k] for k in ('id','artist','title','artwork_key','image_sha256','split')} for aid in sorted(selected)],'controls':controls,'limitations':'No cached DINO embeddings exist; CPU pHash only. Catalog identity and image hashes exact; perceptual distance does not establish distinct underlying artworks. No GPU scoring or generation performed.'}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('exact_duplicate_groups','hash_mismatches','near_duplicate_candidates','closest_cross_split_pairs')},indent=2))

if __name__=='__main__': main()
