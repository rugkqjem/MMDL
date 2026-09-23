import argparse
import ast
import os
import re
import string
import time
from tqdm import tqdm
from datasets import load_dataset 
import pandas as pd
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

def parse_arguments():
    parser=argparse.ArgumentParser(description="MMMU Evaluation Script")
    parser.add_argument("--model_id",type=str,default="Qwen/Qwen3-VL-4B-Instruct",help="Model ID or checkpoint path",)
    parser.add_argument("--revision_id",type=str,default='ebb281ec70b05090aa6165b016eac8ec08e71b17',help="Model revision hash or branch",)
    parser.add_argument("--max-new-tokens",type=int,default=2048,help="Maximum number of tokens to generate(default:2048)")
    parser.add_argument("--temperature",type=float,default=0.7,help="Temperature for sampling(default:0.7 for greedy-like decoding)")
    parser.add_argument("--top-p",type=float,default=0.9,help="Top-p for sampling (default:20 for greedy decoding)")
    parser.add_argument("--top-k",type=int,default=1,help="Top-k for sampling")
    parser.add_argument("--repetition-penalty",type=float,default=1.0,help="Repetition penalty (default:1.0, increase to 1.2-1.5 to reduce repetion)")
    parser.add_argument("--presence-penalty",type=float,default=1.5,help="Presence penalty (default:1.5, range:0.0-2.0,penalize tokens that have already appeared)")
    parser.add_argument("--min_pixels",type=float,default=256*28*28,help="Standard resolution settings")
    parser.add_argument("--max_pixels",type=float,default=1280*28*28,help="Standard resolution settings")
    parser.add_argument("--output_dir",type=str,default="./results",help="Directory to save CSVs")

    return parser.parse_args()

def parse_options(opt_value):
  '''
  if question_type is "multiple-choice": opt_value = "['choice_1','choice_2',...]"
  if question_type i "open" : opt_value ="[]"

  return 
  ("multiple-choice") --> ["choice_1","choice_2",...]
  ("open") --> []
  '''
  if isinstance(opt_value,list):
    return opt_value
  
  if pd.isna(opt_value) or not opt_value or opt_value=='[]':
    return []
  
  try:
    parsed = ast.literal_eval(opt_value)
    if isinstance(parsed,list):
      return parsed
  except (SyntaxError, ValueError):
    pass
  return [] 

def prepare_dataframe(subject,args):
  dataset=load_dataset("MMMU/MMMU",subject,split="validation")

  df=pd.DataFrame(dataset)
  df["subject"]=subject

  messages_list=[]
  options_dict_list=[]


  for idx,sample in df.iterrows():
    question=sample["question"]
    prompt=f'Question:{question}\n'
    options_list=parse_options(sample['options'])
    options_dict={}

    #객관식(multiple-choice) 문제인 경우 
    if sample['question_type']=='multiple-choice' and options_list:
        formatted_options='Options:\n'
        for opt_idx,opt in enumerate(options_list):
            if opt:
                cand=string.ascii_uppercase[opt_idx]
                options_dict[cand]=opt
                formatted_options+=f'{cand}. {opt}\n'
        prompt+=formatted_options
        prompt+="Please select the correct answer from the options above.\n"

        #prompt 응답 형식 강조 추가
        prompt += "\nCRITICAL: Output ONLY the single option letter corresponding to the correct answer. Do NOT provide any reasoning, thinking, or extra text.\nAnswer:"

    #주관식(open) 문제인 경우 추가 prompt 
    else:
       prompt+="Answer the question using a single word or phrase."

       #prompt 응답 형식 강조 추가 
       prompt+="\nAnswer directly with a short text response. Do NOT show thinking.\nAnswer:"
    
    content=[]

    for i in range(1,8):
        img_key=f'image_{i}'
        if img_key in sample and sample[img_key] is not None:
          content.append({"type":"image","image":sample[img_key],"min_pixels":args.min_pixels,"max_pixels":args.max_pixels})

    content.append({"type":"text","text":prompt})
    message=[{"role":"user","content":content}]
    messages_list.append(message)
    options_dict_list.append(options_dict)

  df["messages"]=messages_list
  df["options"]=options_dict_list

  return df
#--------------------------------------------------------------
def parse_multi_choice_resonse(response,all_choices):
   clean_resp=response.strip()
   patterns = [
        r"(?:the\s+)?correct\s+option\s+is\s+[\"\']?([A-Z])[\"\']?", # the correct option is 'A'
        r"(?:the\s+)?correct\s+answer\s+is\s+[\"\']?([A-Z])[\"\']?", # correct answer is B
        r"answer\s*:\s*[\"\']?([A-Z])[\"\']?",                       # answer: C
        r"option\s*:\s*[\"\']?([A-Z])[\"\']?",                       # option: D
        r"\b([A-Z])\b\s*is\s+correct",                               # A is correct
        r"([A-Z])\)",                                                # A)
        r"\(([A-Z])\)",                                              # (A)
    ]
   for pattern in patterns:
      match=re.search(pattern,clean_resp,re.IGNORECASE)
      if match:
         cand=match.group(1).upper()
         if cand in all_choices:
            return cand
   chars=[c.upper() for c in clean_resp if c.upper() in all_choices]
   if len(chars)==1:
      return chars[0]
   for choice in all_choices:
      if choice in clean_resp:
         return choice
   return ""

def parse_open_response(response):
   clean_resp=response.strip().split('\n')[0]
   clean_resp=re.sub(r"[^\w\s\.-]","",clean_resp).strip()
   return clean_resp

def answer_parser(output_text,question_type,options_dict):
   if question_type=="multiple-choice":
      all_choices=list(options_dict.keys())
      if not all_choices:
         all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
      return parse_multi_choice_resonse(output_text,all_choices)
   else:
      return parse_open_response(output_text)
   
#---------------------------------------------------------------

def run_infer_and_eval(model,processor,df,args):
   pred_answers=[]
   raw_outputs=[]
   total_samples=len(df)
   
   print(f"\n[INFO] 총 {total_samples}개 문항에 대한 추론 시작합니다. ")
   for idx,row in tqdm(df.iterrows(),total=total_samples,desc="Inference Progress"):
      start_time=time.time()
      message=row["messages"]
      question_type=row["question_type"]
      options_dict=row["options"]
      q_id=row.get("id",idx)

      inputs=processor.apply_chat_template(
         message,
         tokenize=True,
         add_generation_prompt=True,
         return_dict=True,
         return_tensors="pt"
      )
      inputs=inputs.to(model.device)

      with torch.no_grad():
         generated_ids=model.generate(
            **inputs,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
         )

      in_prompt_len = inputs.input_ids.shape[1]
      generated_ids_trimmed = generated_ids[:, in_prompt_len:]
      output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
      raw_outputs.append(output_text)
      pred=answer_parser(output_text,question_type,options_dict)
      pred_answers.append(pred)
      elapsed=time.time()-start_time
   df['model_raw_output']= raw_outputs
   df['pred_answer']=pred_answers
   df['is_correct']=df.apply(
      lambda r: str(r["pred_answer"]).strip().lower()
      ==str(r["answer"]).strip().lower(),
      axis=1,
   )

   acc=df['is_correct'].mean()*100

   print(" 추론 종료  ")
   return df,acc


def main():

    args=parse_arguments()
    subjects = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    #subjects=["Accounting"]
    model_id=args.model_id
    revision_id=args.revision_id
    summary_results=[]
    all_wrong_dfs=[]

    print(f"[1/3] 모델 및 프로세서 로딩 중...({args.model_id})")
    model=Qwen3VLForConditionalGeneration.from_pretrained(
       model_id,
       revision=revision_id,
       dtype="auto",
       device_map="auto",
       attn_implementation="sdpa"
    )

    processor=AutoProcessor.from_pretrained(model_id,revision=revision_id)
    print("[1/3] 모델 로딩 완료")
    for subject in subjects:
        print(f"\n[2/3] '{subject}' 데이터셋 불러오는 중...")
        dataset=load_dataset("MMMU/MMMU",subject,split='validation')
        print(f"[2/3] '{subject}' 프롬프트 및 이미지 메시지 구성 중 ...")
        prepared_df=prepare_dataframe(subject,args)

        print(f"[3/3] '{subject}' 추론 및 평가 실행 중 ... ")
        df,acc=run_infer_and_eval(
           model=model,
           processor=processor,
           df=prepared_df,
           args=args,
        )

        raw_csv_path=os.path.join(args.output_dir,f"raw_result_{subject}.csv")
        save_df=df.copy()
        if "messages" in save_df.columns:
         save_df=save_df.drop(columns=["messages"])
        save_df.to_csv(raw_csv_path,index=False,encoding="utf-8-sig")
        print(f"-> 현재 subject raw 추론 결과 저장 : {raw_csv_path}")

        summary_results.append(
           {
              "subject":subject,
              "accuracy":round(acc,2),
              "total_questions":len(df),
              "correct_count":int(df["is_correct"].sum()),
              "wrong_count":int((~df["is_correct"]).sum()),
           }
        )
        wrong_df=df[~df["is_correct"]].copy()

        if "messages" in wrong_df.columns:
           wrong_df=wrong_df.drop(columns=["messages"])
        all_wrong_dfs.append(wrong_df)

    summary_df=pd.DataFrame(summary_results)
    mean_acc=summary_df["accuracy"].mean()

    overall_row=pd.DataFrame(
       [
          {
             "subject":"OVERALL_AVERAGE",
             "accuracy":round(mean_acc,2),
             "total_questions":summary_df["total_questions"].sum(),
             "correct_count":summary_df["correct_count"].sum(),
             "wrong_count":summary_df["wrong_count"].sum(),
          }
       ]
    )
    summary_df=pd.concat([summary_df,overall_row],ignore_index=True)
    total_wrong_df=pd.concat(all_wrong_dfs,ignore_index=True)

    if not total_wrong_df.empty:
       priority_cols=["subject","id","question_type","question","options","answer","pred_answer","model_raw_output"]
       other_cols=[c for c in total_wrong_df.columns if c not in priority_cols]
       total_wrong_df=total_wrong_df[priority_cols + other_cols]

    csv_path=os.path.join(args.output_dir,"mmmu_evaluation_report.csv")

    with open(csv_path,"w",encoding="utf-8-sig") as f:
       f.write("===SECTION 1:SUBJECT ACCURACY SUMMARY ===\n")
       summary_df.to_csv(f,index=False)
       f.write("\n=== SECTION 2:WRONG ANSWERS DETAIL ANALYSIS ===\n")
       total_wrong_df.to_csv(f,index=False)

    print(f"\n==============================")
    print(f"Overall Average Accuracy: {mean_acc:.2}%")
    print(f"Single CSV Report saved successfully at:{csv_path}")
    print(f"\n==============================")

if __name__ =="__main__":
   main()

      