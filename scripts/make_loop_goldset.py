"""Build a hand-labeling set of truncated ViRL rollout responses (gold set for the loop-onset detector). CPU only.

Picks truncated responses (finish_reason == length) from the rollouts, one per question, only for questions in the
final training set (data/rl/train_final.jsonl, where the DPO pairs will come from), from two pools:
  - flagged: classify_errors.status() == 'loop' (zlib < 0.10 on the last 10k chars, or >= 80% repeated tail lines)
  - unflagged: truncated but not flagged (would be thrown away as "long calculation" by that rule)
The labeler is not told which pool a sample came from, nor the rule's metrics, so the labels stay independent of the
rule we are checking. Those go to loop_gold_key.csv, to be joined only after labeling.

Outputs (outputs/loop_gold/):
  loop_gold.html     open in a browser: question, gold answer, numbered response rows; click a row to mark
                     the repeat onset / branch point, pick the loop type, then "CSV 내보내기" -> loop_gold_labels.csv
  loop_gold_key.csv  pool, zlib ratio, repeated-line ratio, tokens per sample (do not open before labeling)

  python scripts/make_loop_goldset.py                    # final training set, 20 flagged + 10 unflagged, seed 0
  python scripts/make_loop_goldset.py --n_flagged 30 --n_unflagged 15 --seed 1
Rows: the response split at newlines, lines longer than --row_chars cut into pieces; each row keeps its char offset,
so labels are char positions comparable with a detector's output.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_errors import repeated_line_ratio, zlib_ratio  # noqa: E402

LOOP_TYPES = [
    ('A1', 'A1 · 계산값이 보기와 안 맞아 결론 못 냄'),
    ('A2', 'A2 · 계산값이 보기와 안 맞아 결론 못 냄'),
    ('C', 'C · 반복을 그 자리에서 끊지 못함 (디코딩 습관)'),
    ('F', 'F · 반복을 그 자리에서 끊지 못함 (디코딩 습관)'),
    ('E', 'E · 이미지를 못 읽거나 확신이 없음'),
    ('D', 'D · 이미지를 못 읽거나 확신이 없음'),
    ('other', '기타'),
]


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def is_flagged(text):
    return zlib_ratio(text) < 0.10 or repeated_line_ratio(text) >= 0.8


def to_rows(text, row_chars):
    rows, pos = [], 0
    for line in text.split('\n'):
        if not line:
            rows.append((pos, ''))
        for i in range(0, len(line), row_chars):
            rows.append((pos + i, line[i:i + row_chars]))
        pos += len(line) + 1
    return rows


def pick(cands, n, rng):
    """One sample per question, spread over categories (round-robin over shuffled categories)."""
    by_cat = defaultdict(list)
    for c in cands:
        by_cat[c['category']].append(c)
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    out, seen = [], set()
    while len(out) < n and any(by_cat.values()):
        for cat in cats:
            while by_cat[cat]:
                c = by_cat[cat].pop()
                if c['id'] not in seen:
                    seen.add(c['id'])
                    out.append(c)
                    break
            if len(out) == n:
                break
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--resp', nargs='+', default=['outputs/rl_rollout/rollouts_resp.jsonl',
                                                 'outputs/rl_rollout/pilot_resp.jsonl'])
    p.add_argument('--rollouts', default='outputs/rl_rollout/rollouts.jsonl', help='category per question')
    p.add_argument('--questions', default='outputs/decontam/virl/clean.jsonl', help='question text, options')
    p.add_argument('--ids', nargs='*', default=['data/rl/train_final.jsonl'],
                   help='only questions listed in these JSONL files (default: final GRPO/DPO training set); '
                        'pass --ids with no file to use every rollout question')
    p.add_argument('--out_dir', default='outputs/loop_gold')
    p.add_argument('--n_flagged', type=int, default=20)
    p.add_argument('--n_unflagged', type=int, default=10)
    p.add_argument('--row_chars', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    cat = {r['id']: r.get('category', '') for r in read_jsonl(a.rollouts)}
    qs = {r['id']: r for r in read_jsonl(a.questions)}
    keep = set().union(*({r['id'] for r in read_jsonl(f)} for f in a.ids)) if a.ids else None
    pools = {'flagged': [], 'unflagged': []}
    n_trunc = 0
    for path in a.resp:
        for g in read_jsonl(path):
            for k, s in enumerate(g['samples']):
                if s['finish_reason'] != 'length' or g['id'] not in qs or (keep is not None and g['id'] not in keep):
                    continue
                n_trunc += 1
                rec = {'id': g['id'], 'sample': k, 'category': cat.get(g['id'], qs[g['id']].get('category', '')),
                       'answer': g['answer'], 'options': g['options'], 'response': s['response'],
                       'tokens': s.get('num_output_tokens'), 'correct_eval': s.get('correct_eval')}
                pools['flagged' if is_flagged(s['response']) else 'unflagged'].append(rec)
    rng = random.Random(a.seed)
    chosen = [dict(c, pool='flagged') for c in pick(pools['flagged'], a.n_flagged, rng)]
    chosen += [dict(c, pool='unflagged') for c in pick(pools['unflagged'], a.n_unflagged, rng)]
    rng.shuffle(chosen)  # blind: pool order not visible
    print(f'questions: {"all rollouts" if keep is None else ", ".join(a.ids)} | truncated {n_trunc}: flagged {len(pools["flagged"])}, unflagged {len(pools["unflagged"])} '
          f'-> picked {sum(c["pool"] == "flagged" for c in chosen)} + {sum(c["pool"] == "unflagged" for c in chosen)}')

    os.makedirs(a.out_dir, exist_ok=True)
    items = []
    with open(os.path.join(a.out_dir, 'loop_gold_key.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['key', 'id', 'sample', 'pool', 'zlib_last10k', 'repeated_line_ratio', 'tokens', 'chars',
                    'correct_eval'])
        for i, c in enumerate(chosen):
            key = f'G{i + 1:02d}'
            w.writerow([key, c['id'], c['sample'], c['pool'], round(zlib_ratio(c['response']), 4),
                        round(repeated_line_ratio(c['response']), 3), c['tokens'], len(c['response']),
                        c['correct_eval']])
            q = qs[c['id']]
            items.append({'key': key, 'id': c['id'], 'sample': c['sample'], 'category': c['category'],
                          'question': q['conversations'][0]['value'], 'answer': c['answer'],
                          'image': q.get('image'), 'chars': len(c['response']),
                          'rows': to_rows(c['response'], a.row_chars)})
    page = PAGE.replace('__TYPES__', json.dumps(LOOP_TYPES, ensure_ascii=False)) \
               .replace('__DATA__', json.dumps(items, ensure_ascii=False).replace('</', '<\\/')) \
               .replace('__TITLE__', html.escape(f'루프 정답지 ({len(items)}개, seed {a.seed})')) \
               .replace('__STORE__', 'loop_gold_' + hashlib.md5(''.join(i['id'] + str(i['sample']) for i in items)
                                                             .encode()).hexdigest()[:10])
    with open(os.path.join(a.out_dir, 'loop_gold.html'), 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'wrote {a.out_dir}/loop_gold.html and loop_gold_key.csv')


PAGE = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e3e2dc;--card:#fff;--onset:#fde2c8;--branch:#d8e6fb;--acc:#b5532a}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecebe6;--mut:#9a9993;--line:#34332f;--card:#22221f;--onset:#5a3a1f;--branch:#203552;--acc:#e39066}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0 8px 0 0}button,select,input,textarea{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{border-color:var(--acc);color:var(--acc);font-weight:600}
main{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px}@media(max-width:860px){main{grid-template-columns:1fr}}
aside{position:sticky;top:64px;align-self:start;display:flex;flex-direction:column;gap:10px;max-height:calc(100vh - 80px);overflow:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.mut{color:var(--mut);font-size:12px}pre.q{white-space:pre-wrap;margin:6px 0 0;font:13px/1.45 ui-monospace,Menlo,monospace}
label{display:block;font-size:12px;color:var(--mut);margin-top:8px}select,input,textarea{width:100%}
#resp{font:12.5px/1.5 ui-monospace,Menlo,monospace}.r{display:grid;grid-template-columns:56px 1fr;cursor:pointer;border-radius:3px}
.r:hover{background:var(--line)}.r .n{color:var(--mut);text-align:right;padding-right:10px;user-select:none}.r .t{white-space:pre-wrap;word-break:break-all}
.r.onset{background:var(--onset)}.r.branch{background:var(--branch)}.r.after{opacity:.55}
.dots span{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--line);margin:1px;cursor:pointer}.dots span.done{background:var(--acc)}.dots span.cur{outline:2px solid var(--fg)}
</style></head><body>
<header><h1>__TITLE__</h1><button id="prev">← 이전</button><span id="pos"></span><button id="next">다음 →</button>
<span class="dots" id="dots"></span><span style="flex:1"></span><button id="export">CSV 내보내기</button></header>
<main><aside>
 <div class="card"><div id="meta" class="mut"></div><pre class="q" id="question"></pre><div class="mut" id="gold"></div></div>
 <div class="card">
  <div class="mut">줄을 클릭하면 아래에서 고른 표시가 찍혀요. 같은 줄을 다시 누르면 지워져요.</div>
  <div style="display:flex;gap:6px;margin-top:6px"><button id="m_onset" class="on">반복 시작</button><button id="m_branch">갈림길 (선택)</button></div>
  <label>루프인가?</label><select id="is_loop"><option value="">—</option><option value="loop">루프 (이미 나온 내용을 다시 씀)</option><option value="no">루프 아님 (계속 새 계산)</option><option value="unsure">애매</option></select>
  <label>루프 유형</label><select id="loop_type"></select>
  <label>반복 시작 = 이미 앞에서 나온 내용이 다시 시작되는 첫 줄</label><input id="onset" readonly placeholder="줄을 클릭">
  <label>갈림길 = 여기서 잘못 접어들었다고 보는 줄 (선택)</label><input id="branch" readonly placeholder="줄을 클릭">
  <label>메모</label><textarea id="note" rows="3"></textarea>
 </div>
</aside><section class="card" id="resp"></section></main>
<script>
const TYPES=__TYPES__, DATA=__DATA__, STORE='__STORE__';  // per sample set: a regenerated page never reuses old labels
let labels={}; try{labels=JSON.parse(localStorage.getItem(STORE)||'{}')}catch(e){}
let cur=0, mode='onset';
const $=id=>document.getElementById(id);
$('loop_type').innerHTML='<option value="">—</option>'+TYPES.map(([v,t])=>`<option value="${v}">${t}</option>`).join('');
function save(){try{localStorage.setItem(STORE,JSON.stringify(labels))}catch(e){} dots()}
function L(){const k=DATA[cur].key;return labels[k]||(labels[k]={})}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function dots(){$('dots').innerHTML=DATA.map((d,i)=>`<span title="${d.key}" class="${labels[d.key]&&labels[d.key].is_loop?'done':''} ${i===cur?'cur':''}" data-i="${i}"></span>`).join('')}
function render(){
 const d=DATA[cur], l=L();
 $('pos').textContent=`${d.key}  (${cur+1}/${DATA.length})`;
 $('meta').textContent=`${d.category} · ${d.id} · sample ${d.sample} · ${d.chars.toLocaleString()}자`;
 $('question').textContent=d.question; $('gold').textContent='정답: '+d.answer+(d.image?'  · 이미지: '+d.image.join(', '):'');
 for(const f of ['is_loop','loop_type','note']) $(f).value=l[f]||'';
 $('onset').value=l.onset_row?`줄 ${l.onset_row} (문자 ${l.onset_char})`:''; $('branch').value=l.branch_row?`줄 ${l.branch_row} (문자 ${l.branch_char})`:'';
 $('resp').innerHTML=d.rows.map(([off,t],i)=>{const n=i+1;let c='r';if(n===l.onset_row)c+=' onset';else if(n===l.branch_row)c+=' branch';else if(l.onset_row&&n>l.onset_row)c+=' after';
   return `<div class="${c}" data-n="${n}" data-off="${off}"><span class="n">${n}</span><span class="t">${esc(t)||' '}</span></div>`}).join('');
 dots();
}
$('resp').addEventListener('click',e=>{const r=e.target.closest('.r');if(!r)return;const n=+r.dataset.n,off=+r.dataset.off,l=L(),k=mode+'_row';
 if(l[k]===n){delete l[k];delete l[mode+'_char']}else{l[k]=n;l[mode+'_char']=off} save(); const y=scrollY; render(); scrollTo(0,y)});
for(const f of ['is_loop','loop_type','note']) $(f).addEventListener('input',()=>{L()[f]=$(f).value;save()});
$('m_onset').onclick=()=>{mode='onset';$('m_onset').classList.add('on');$('m_branch').classList.remove('on')};
$('m_branch').onclick=()=>{mode='branch';$('m_branch').classList.add('on');$('m_onset').classList.remove('on')};
$('prev').onclick=()=>{cur=(cur+DATA.length-1)%DATA.length;render();scrollTo(0,0)};
$('next').onclick=()=>{cur=(cur+1)%DATA.length;render();scrollTo(0,0)};
$('dots').onclick=e=>{if(e.target.dataset.i){cur=+e.target.dataset.i;render();scrollTo(0,0)}};
$('export').onclick=()=>{const cols=['key','id','sample','category','chars','is_loop','loop_type','onset_row','onset_char','branch_row','branch_char','note'];
 const q=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
 const lines=[cols.join(',')].concat(DATA.map(d=>{const l=labels[d.key]||{};return cols.map(c=>q(c in d?d[c]:l[c])).join(',')}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+lines.join('\n')],{type:'text/csv'}));a.download='loop_gold_labels.csv';a.click()};
render();
</script></body></html>'''

if __name__ == '__main__':
    main()
