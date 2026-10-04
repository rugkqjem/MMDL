# Sanity check of train_ft.py's labels / collator on the real model: loss of the model's own greedy answer must be ~0.
import sys, json, torch
sys.path.insert(0, '/workspace/MMDL/scripts')
import train_ft as T
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
P = 'Qwen/Qwen3-VL-4B-Instruct'
proc = AutoProcessor.from_pretrained(P, revision=T.MODEL_REVISION)
proc.image_processor.size = {'shortest_edge': 64 * 32 * 32, 'longest_edge': 256 * 32 * 32}
model = Qwen3VLForConditionalGeneration.from_pretrained(P, revision=T.MODEL_REVISION, dtype=torch.bfloat16).cuda().eval()
row = T.load_rows('/workspace/MMDL/data/smoke/train.jsonl')[0]
msgs = T.to_messages(row, '/workspace/MMDL/data/smoke/images')
enc = proc.apply_chat_template(msgs[:1], tokenize=True, return_dict=True, return_tensors='pt', add_generation_prompt=True).to('cuda')
out = model.generate(**enc, max_new_tokens=48, do_sample=False)
greedy = proc.tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
print('GREEDY:', repr(greedy))
ds = T.SFTDataset([], proc, '/workspace/MMDL/data/smoke/images', 8192)
coll = T.make_collator(proc.tokenizer.pad_token_id)
for name, target in [('own greedy answer', greedy), ('dataset target', row['conversations'][1]['value'])]:
    r = dict(row, conversations=[row['conversations'][0], {'from': 'gpt', 'value': target}])
    ds.rows = [r]
    b = {k: v.cuda() for k, v in coll([ds[0]]).items()}
    with torch.no_grad():
        o = model(**b)
    lab = b['labels'][0, 1:]; lp = torch.log_softmax(o.logits[0, :-1].float(), -1)
    m = lab != -100
    tok = lp[m].gather(1, lab[m, None])[:, 0]
    print(f'{name}: loss={o.loss.item():.3f} n_tokens={int(m.sum())} per-token nll=',
          [(proc.tokenizer.decode([int(t)]), round(-float(x), 2)) for t, x in zip(lab[m], tok)][:12])
