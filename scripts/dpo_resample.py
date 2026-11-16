"""Re-generate from branch points of looping responses: does the model escape from there? (GPU, vLLM)

For each truncated response in loop_candidates.jsonl (loop_onset.py), cut it at a branch point, give
  eval prompt (infer.prepare_sample, same text and image resize as MMMU eval) + response[:branch]
to vLLM as the assistant's partial answer, and sample k continuations with the eval sampling recipe and the
original budget (8192 tokens in total, prefix included). A continuation "escapes" when it ends with EOS.
Branch points (60/45/30/15 % of the response, from loop_onset.py) are tried latest first; a response moves to the
next (earlier) one only if no continuation escaped, so the kept branch is the latest point where the model still
gets out. This replaces locating the loop onset in text, which the gold set showed to be unreliable.

Run with the model DPO will start from (the RFT/SFT checkpoint): the chosen continuations must be that model's own
outputs, and DPO's reference model is that same checkpoint.

  python scripts/dpo_resample.py --model_path outputs/rft/merged --out outputs/dpo/resample.jsonl
  python scripts/dpo_resample.py --model_path Qwen/Qwen3-VL-4B-Instruct --limit 20      # smoke test, base model
Output, one row per truncated response: id, sample, dense_char, tried [{branch_char, n, escaped, correct}],
branch_char (kept, or null), prefix_tokens, continuations [{text, finish_reason, tokens, correct}] at the kept branch.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer  # noqa: E402
from rl_reward import score_response  # noqa: E402


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_responses(cands, keep_ids, per_question, max_branches, only_flagged=False):
    """Truncated responses to resample, at most per_question per question. All truncated responses by default
    (the gold set found every labeled one to be a loop); only_flagged keeps those a text rule calls a loop."""
    by_q = defaultdict(list)
    for c in cands:
        if c['id'] in keep_ids and c.get('branches') and (not only_flagged or c['loop'] or c['old_rule_loop']):
            by_q[c['id']].append(c)
    out = []
    for q in sorted(by_q):
        for c in sorted(by_q[q], key=lambda c: (not (c['loop'] or c['old_rule_loop']), c['sample']))[:per_question]:
            out.append(dict(c, branches=c['branches'][:max_branches]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--candidates', default='outputs/dpo/loop_candidates.jsonl')
    p.add_argument('--resp', nargs='+', default=['outputs/rl_rollout/rollouts_resp.jsonl',
                                                 'outputs/rl_rollout/pilot_resp.jsonl'])
    p.add_argument('--questions', default='outputs/decontam/virl/clean.jsonl')
    p.add_argument('--ids', default='data/rl/train_final.jsonl')
    p.add_argument('--image_root', default='data/virl39k_raw')
    p.add_argument('--out', default='outputs/dpo/resample.jsonl')
    p.add_argument('--k', type=int, default=8, help='continuations per branch point')
    p.add_argument('--per_question', type=int, default=2, help='truncated responses per question')
    p.add_argument('--only_flagged', action='store_true', help='skip truncations no text rule calls a loop')
    p.add_argument('--max_branches', type=int, default=4)
    p.add_argument('--total_tokens', type=int, default=8192, help='prefix + continuation, as in the rollouts')
    p.add_argument('--min_continuation', type=int, default=512, help='skip branches leaving fewer tokens')
    p.add_argument('--limit', type=int, default=None, help='first N responses (smoke test)')
    p.add_argument('--batch', type=int, default=64, help='prompts per vLLM call')
    # model / sampling: eval defaults (infer.py), overridable the same way
    p.add_argument('--model_path', required=True)
    p.add_argument('--model_revision', default=None, help='default: infer.MODEL_REVISION for the hub base model')
    p.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)

    ev = infer.parse_args(['--output', os.devnull])           # eval recipe: pixels, temperature, penalties
    ev.model_path = a.model_path
    ev.model_revision = (a.model_revision if a.model_revision is not None else
                         (infer.MODEL_REVISION if not os.path.isdir(a.model_path) else ''))
    ev.max_model_len = 32768
    ev.gpu_memory_utilization = a.gpu_memory_utilization

    keep = {r['id'] for r in read_jsonl(a.ids)}
    todo = pick_responses(read_jsonl(a.candidates), keep, a.per_question, a.max_branches, a.only_flagged)[:a.limit]
    need = {(c['id'], c['sample']) for c in todo}
    resp, gold = {}, {}
    for path in a.resp:
        for g in read_jsonl(path):
            for k, s in enumerate(g['samples']):
                if (g['id'], k) in need:
                    resp[(g['id'], k)] = s['response']
                    gold[g['id']] = (g['answer'], g['options'])
    rows = {r['id']: r for r in read_jsonl(a.questions) if r['id'] in {c['id'] for c in todo}}
    print(f'[resample] {len(todo)} truncated responses from {len({c["id"] for c in todo})} questions, '
          f'k={a.k}, up to {a.max_branches} branch points each', flush=True)

    from transformers import AutoProcessor
    from rl_rollout_filter import to_sample
    processor = AutoProcessor.from_pretrained(a.model_path, revision=ev.model_revision or None)
    tok = processor.tokenizer
    llm = infer.make_llm(ev)
    base_inputs = {}
    for q in rows:
        inp, _ = infer.prepare_sample(to_sample(rows[q], a.image_root), rows[q].get('category', ''), processor, ev)
        base_inputs[q] = inp

    state = {(c['id'], c['sample']): {'id': c['id'], 'sample': c['sample'], 'dense_char': c.get('dense_char'),
                                      'branches': c['branches'], 'tried': [], 'branch_char': None,
                                      'prefix_tokens': None, 'continuations': []} for c in todo}
    t0 = time.time()
    for rnd in range(a.max_branches):
        jobs = []
        for key, st in state.items():
            if st['branch_char'] is not None or rnd >= len(st['branches']):
                continue
            if st['tried'] and st['tried'][-1]['escaped'] > 0:
                continue
            b = st['branches'][rnd]
            prefix = resp[key][:b]
            n_prefix = len(tok(prefix, add_special_tokens=False)['input_ids'])
            budget = a.total_tokens - n_prefix
            if budget < a.min_continuation:
                st['tried'].append({'branch_char': b, 'n': 0, 'escaped': 0, 'correct': 0, 'skipped': 'no budget'})
                continue
            inp = dict(base_inputs[st['id']], prompt=base_inputs[st['id']]['prompt'] + prefix)
            jobs.append((key, b, prefix, n_prefix, budget, inp))
        if not jobs:
            break
        print(f'[round {rnd + 1}] {len(jobs)} prefixes x {a.k}', flush=True)
        for i in range(0, len(jobs), a.batch):
            chunk = jobs[i:i + a.batch]
            # one SamplingParams per prompt: the continuation budget depends on the prefix length
            sps = []
            for _, _, _, _, budget, _ in chunk:
                ev.max_new_tokens = budget
                sps.append(infer.make_sampling(ev, n=a.k))
            outs = llm.generate([j[5] for j in chunk], sampling_params=sps, use_tqdm=False)
            for (key, b, prefix, n_prefix, budget, _), out in zip(chunk, outs):
                ans, opts = gold[key[0]]
                conts = []
                for o in out.outputs:
                    stop = o.finish_reason == 'stop'
                    sc = score_response(prefix + o.text, ans, opts, truncated=not stop)
                    conts.append({'text': o.text, 'finish_reason': o.finish_reason, 'tokens': len(o.token_ids),
                                  'correct': sc['correct'], 'parsed': sc['parsed']})
                esc = sum(c['finish_reason'] == 'stop' for c in conts)
                st = state[key]
                st['tried'].append({'branch_char': b, 'n': len(conts), 'escaped': esc,
                                    'correct': sum(c['correct'] for c in conts)})
                if esc:
                    st.update(branch_char=b, prefix_tokens=n_prefix, continuations=conts)
            print(f'  {min(i + a.batch, len(jobs))}/{len(jobs)} ({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for st in state.values():
            st.pop('branches')
            f.write(json.dumps(st, ensure_ascii=False) + '\n')
    kept = [s for s in state.values() if s['branch_char'] is not None]
    n_esc = sum(t['escaped'] for s in kept for t in s['tried'][-1:])
    print(f'[resample] escaped at some branch: {len(kept)}/{len(state)} responses; '
          f'escaping continuations at kept branches {n_esc}; wrote {a.out} ({time.time() - t0:.0f}s)')
    with open(os.path.splitext(a.out)[0] + '_meta.json', 'w') as f:
        json.dump(vars(a) | {'sampling': {'temperature': ev.temperature, 'top_p': ev.top_p, 'top_k': ev.top_k,
                                          'presence_penalty': ev.presence_penalty,
                                          'repetition_penalty': ev.repetition_penalty},
                             'model_revision_used': ev.model_revision}, f, indent=2)


if __name__ == '__main__':
    main()
