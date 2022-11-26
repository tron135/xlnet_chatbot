import json
from transformers import XLNetTokenizer, XLNetForQuestionAnsweringSimple
from torch.nn import functional as F
import torch
import time

x = open("dataset.json")


# import the demo JSON dataset
data = json.load(x)['data']


def get_subjects():
    for i in range(len(data)):
        subject = data[i]['subject_name']
        print(f'{i+1}. {subject}')


def print_topics(data):
    for i in range(len(data)):
        topic = data[i]['topic_name']
        print(f'\n{i+1}. {topic}')


def get_description(top_num, data):
    global continue_param
    global context
    if type(data['description']) == list:
        print_topics(data['description'])
        topic_num = int(input('\n\n Which topic you want help with ? \n\n'))
        get_description(topic_num, data['description'][topic_num - 1])
    else:
        continue_param = False
        context = data['description']
        topic_name = data['topic_name']
        print(f'{topic_name}:')
        print(context)


# Initializing tokenizer and model from fine-tuned model folder
tokenizer = XLNetTokenizer.from_pretrained("fine-tuned-model")
model = XLNetForQuestionAnsweringSimple.from_pretrained("fine-tuned-model", return_dict = True)


# getting the answer using the fine-tuned model
def get_answer(que, text):
    inputs = tokenizer.encode_plus(que, text, return_tensors='pt')
    output = model(**inputs)
    start_max = torch.argmax(F.softmax(output.start_logits, dim = -1))
    end_max = torch.argmax(F.softmax(output.end_logits, dim=-1)) + 1 
    
    answer = tokenizer.decode(inputs["input_ids"][0][start_max : end_max])
    print(answer)


continue_param = True
context = ""

print('The list of subjects available: \n')

# Getting list of subjects
get_subjects()
time.sleep(1)

# Getting input from student
subject_num = int(input('\n\n In which subject can I help you today !? \n\n'))


subject_name = data[subject_num-1]['subject_name']

print(f'The list of topics available in {subject_name}')

# Getting list of topics
print_topics(data[subject_num - 1]['topics'])
time.sleep(1)

# Getting input from student
topic_num = int(input('\n\n Which topic you want help with ? \n\n'))
time.sleep(1)

while continue_param == True:
    get_description(topic_num, data[subject_num - 1]['topics'][topic_num - 1])

question = str(input('\n\n What is your question ? \n\n'))

get_answer(question, context)




