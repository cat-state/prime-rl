"""Training-only format/completion diagnostics, no policy or reward modifications."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def stats(values):
    return {'n':len(values),'min':int(min(values)),'median':float(np.median(values)),'p90':float(np.quantile(values,.9)),'max':int(max(values))} if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument('episodes',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
    jobs={j['seed']:j for j in json.loads((a.episodes/'jobs.json').read_text())};rows=[]
    for path in sorted(a.episodes.glob('ep-*/episode.json')):
        state=json.loads(path.read_text());assert state['complete']
        for turn in state['turns']:
            d=path.parent/('turn-'+str(turn['turn']));r=json.loads((d/'response.json').read_text());text=r['raw_output'];marker='<|content_text|>';has=marker in text
            before,after=text.split(marker,1) if has else (text,'');source=(d/'action.wgsl').read_text() if (d/'action.wgsl').exists() else None
            error=turn.get('error','');kind='valid'
            if not turn['valid']:
                if r['finish_reason']=='length':kind='truncated_in_final' if has else 'truncated_before_final'
                elif 'exactly one final WGSL' in error:kind='final_format'
                elif 'immutable' in error or 'invalid left-hand side' in error:kind='immutable_assignment'
                elif 'unknown identifier' in error or 'no definition in scope' in error:kind='unknown_identifier'
                elif 'type' in error.lower():kind='type_or_validation'
                else:kind='other_compile_or_contract'
            rows.append({'episode':str(path.parent.resolve()),'seed':state['seed'],'artwork_id':jobs[state['seed']]['artwork_id'],'turn':turn['turn'],'valid':turn['valid'],'completion_tokens':len(r['completion_ids']),'finish_reason':r['finish_reason'],'has_final_marker':has,'prefinal_characters':len(before),'final_characters':len(after),'reasoning_fraction_character_proxy':len(before)/max(1,len(text)),'source_lines':len(source.splitlines()) if source else None,'source_bytes':len(source.encode()) if source else None,'source':str(d/'action.wgsl') if source else None,'failure_category':kind,'error':error[:700]})
    final=[r for r in rows if r['turn']==2];valid=[r for r in rows if r['valid']];truncated=[r for r in rows if r['finish_reason']=='length']
    summary={'turn_count':len(rows),'failure_counts':dict(Counter(r['failure_category'] for r in rows)),'final_failure_counts':dict(Counter(r['failure_category'] for r in final)),'valid_source_lines':stats([r['source_lines'] for r in valid]),'valid_source_bytes':stats([r['source_bytes'] for r in valid]),'valid_completion_tokens':stats([r['completion_tokens'] for r in valid]),'valid_prefinal_character_fraction_median':float(np.median([r['reasoning_fraction_character_proxy'] for r in valid])),'truncated_before_final_count':sum(not r['has_final_marker'] for r in truncated),'truncated_with_final_count':sum(r['has_final_marker'] for r in truncated),'valid_programs_over40lines':sum(r['source_lines']>40 for r in valid),'char_proxy_warning':'Character fraction is not a token fraction; prefinal includes role/control markers plus reasoning.'}
    (a.output/'audit.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
