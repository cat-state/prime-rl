"""Predeclared corpus-wide border sensitivity; never modifies the RL metric."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.painting_dataset.perceptual_reward import SpatialPerceptual, MODEL, REVISION, LAYERS
from tools.painting_dataset.review_reward import load


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    spec={'model':MODEL,'revision':REVISION,'layers':LAYERS,'crop':'Remove floor(width*.05) on each horizontal side and floor(height*.05) on each vertical side from max256 RGB target; preserves approximately90% width/height. Same crop applies to reference and candidate.','controls':['identity','mean_wash','shuffled_pixels','border_copy_mean_wash','border_copy_shuffled_pixels'],'selection':'All100 downloaded museum targets, no selection by score/frame presence.','interpretation':'Copied border can contain real content, frame, or mat. Sensitivity is not an automatic frame detector. Cropped scores are diagnostic only; production metric remains unchanged.'}
    (args.output/'predeclared-protocol.json').write_text(json.dumps(spec,indent=2)+'\n')
    metric=SpatialPerceptual();records=[]
    rows=[r for r in json.loads(args.manifest.read_text())['records'] if r.get('image')];assert len(rows)==100
    for i,r in enumerate(rows):
        target=load(r['image']);pixels=np.asarray(target);h,w=pixels.shape[:2];dx,dy=int(w*.05),int(h*.05);box=(dx,dy,w-dx,h-dy)
        wash=Image.new('RGB',target.size,tuple(np.rint(pixels.mean((0,1))).astype(int)))
        seed=int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed)
        noise=Image.fromarray(rng.permutation(pixels.reshape(-1,3)).reshape(pixels.shape))
        copied_wash=target.copy();copied_wash.paste(wash.crop(box),(dx,dy))
        copied_noise=target.copy();copied_noise.paste(noise.crop(box),(dx,dy))
        images=[target,wash,noise,copied_wash,copied_noise];names=spec['controls']
        full=dict(zip(names,metric.compare(images),strict=True));interior=dict(zip(names,metric.compare([im.crop(box) for im in images]),strict=True))
        gain={kind:full['border_copy_'+kind]['spatial_cosine']-full[kind]['spatial_cosine'] for kind in ['mean_wash','shuffled_pixels']}
        check=max(abs(interior['border_copy_'+kind]['spatial_cosine']-interior[kind]['spatial_cosine']) for kind in ['mean_wash','shuffled_pixels'])
        assert check<1e-5,check
        records.append({'id':r['id'],'split':r['split'],'title':r['title'],'artist':r['artist'],'image_sha256':r['image_sha256'],'crop_box':box,'border_area_fraction':1-(w-2*dx)*(h-2*dy)/(w*h),'full':full,'interior90':interior,'border_copy_gain':gain,'interior_equivalence_error':check})
        print(str(i+1)+'/100 '+r['id'],flush=True)
    summary={}
    for kind in ['mean_wash','shuffled_pixels']:
        values=np.array([r['border_copy_gain'][kind] for r in records]);summary[kind]={'mean':float(values.mean()),'median':float(np.median(values)),'min':float(values.min()),'max':float(values.max()),'positive_count':int((values>1e-5).sum()),'above_0_05_count':int((values>.05).sum()),'above_0_1_count':int((values>.1).sum())}
    report={'protocol':spec,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'metric_source_sha256':hashlib.sha256(Path(__file__).with_name('perceptual_reward.py').read_bytes()).hexdigest(),'n':len(records),'summary':summary,'records':records}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
