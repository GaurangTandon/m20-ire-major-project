# %%
from summarizer import Summarizer
from pipelines import pipeline
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import sys

# %%
tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', return_dict=True)
# %%
def getdata(filename):
    data=""
    with open(filepath,'r') as fp:
        lines=fp.readlines()
        for line in lines:
           data+=line
    fp.close()
    return data
# %%
# Here it is summarized for cleaned_coa.txt and cleaned_nlp.txt

body=getdata(sys.argv[1])


# %%
# General Bert Summarizer
model = Summarizer()
result = model(body, min_length=60)
# %%
# Code To Generate Question From Text used code from https://github.com/patil-suraj/question_generation.git 
nlp = pipeline("question-generation")

# %%
outfile=open(sys.argv[2],'w')
sentences = result.split("\n")

for sent in sentences:
    if len(sent) == 0:
        continue
    try:
        res = nlp(sent)
    except ValueError:
        continue

    for x in res:
        q, a = x["question"], x["answer"]

        # Bert Model to predict answer given question
        input_ids = tokenizer.encode(q, a)
        sep_index = input_ids.index(tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        start_scores, end_scores = model(torch.tensor([input_ids]),token_type_ids=torch.tensor([segment_ids]))
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        predicted_answer = ' '.join(tokens[answer_start:answer_end+1])
        outfile.write(q+'\n'+predicted_answer)
       


# %%


# %%
# %%
# %%

