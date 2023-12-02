
# ChatGPT - Chatanywhere

import openai
openai.api_base = "https://api.chatanywhere.cn/v1"
openai.api_key = 'sk-0CyrB5s2uouW7QlDqK03c1Quh4dPzK6dqRSW0f42Y6xRJBYE'


# TODO: assistant指定多轮对话，如包含多轮对话，则msgs = [system, user, assistant, user]
def chatWithGPT(system = '', user = '', assistant = '', model = "gpt-3.5-turbo", temperature = 0.7):
    assert isinstance(system, str), '`system` should be a string'
    assert isinstance(user, str), '`user` should be a string'
    assert isinstance(assistant, str), '`system` should be a string'
    msgs = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user}
    ]
    if assistant != '':
        msgs.append({'role': 'assistant', 'content': assistant})
    # sys.stderr.write('[INFO]request msgs: ' + json.dumps(msgs, ensure_ascii = False) + '\n')
    
    response = openai.ChatCompletion.create(model = model,
                                            messages = msgs,
                                            temperature = temperature
                                            )
    status_code = response["choices"][0]["finish_reason"]
    assert status_code == "stop", f"[ERROR]The status code was {status_code}."
    print('GPT response content: ' + response["choices"][0]["message"]["content"])
    return response

def getCommentsWithGPT(text, temperature, model = 'gpt-3.5-turbo'):
    response = chatWithGPT(
        user = f'请重写下面的文本：\n{text}',
        model = model,
        temperature = temperature,
    )
    response_text = response["choices"][0]["message"]["content"]
    return response_text

def rewriteWithGPT(text, temperature, model = 'gpt-3.5-turbo'):
    response = chatWithGPT(
        user = f'请重写下面的文本：\n{text}',
        model = model,
        temperature = temperature,
    )
    response_text = response["choices"][0]["message"]["content"]
    return response_text
# rewriteWithGPT('天阶夜色凉如水，卧看牵牛织女星。', temperature = 0.7, model = 'gpt-3.5-turbo')

def summaryWithGPT(text, temperature, model = 'gpt-3.5-turbo'):
    response = chatWithGPT(
        user = f'请总结下面的文本：\n{text}',
        model = model,
        temperature = temperature,
    )
    response_text = response["choices"][0]["message"]["content"]
    return response_text

import time
def enhanceData(data_list, enhance_model = 'gpt-3.5-turbo'):
    for item in data_list:
        print(item)
        comment = item['comment']
        try:
            rewrite = rewriteWithGPT(comment, temperature = 0.7, model = enhance_model)
            summary = summaryWithGPT(comment, temperature = 0.7, model = enhance_model)
        except:
            time.sleep(1)
            rewrite = rewriteWithGPT(comment, temperature = 0.7, model = enhance_model)
            summary = summaryWithGPT(comment, temperature = 0.7, model = enhance_model)
        item['rewrite'] = rewrite
        item['summary'] = summary
        item['rewrite_model'] = enhance_model
        item['summary_model'] = enhance_model
        print(item)
        time.sleep(0.5)
    return data_list


import json
human_comment_list = json.load(open('ExampleData/Comment/select_human_comment_list.json', 'r'))
GPT_comment_list = json.load(open('ExampleData/Comment/GPT_comment_list.json', 'r'))
GPT4_comment_list = json.load(open('ExampleData/Comment/GPT4_comment_list.json', 'r'))
# GLM_comment_list = json.load(open('ExampleData/Comment/GLM_comment_list.json', 'r'))
print(f'human: {len(human_comment_list)}, GPT: {len(GPT_comment_list)}, GPT4: {len(GPT4_comment_list)}')

# load old
enhanced_human_comment_list = []
with open('log/enhance_log', 'r') as f:
    for line in f:
        if line.startswith("{'comment':"):
            if line.find('"') != -1:
                data = json.loads(line.strip().replace("\"", "$").replace("'", "\""))
                data['comment'] = data['comment'].replace("$", "\"")
            else:
                data = json.loads(line.strip().replace("'", "\""))
            if data['label'] == 'human':
                enhanced_human_comment_list.append(data)
data_len = len(enhanced_human_comment_list)
# json.dump(enhanced_human_comment_list, open(f'ExampleData/Comment/enhanced_human_comment_list_{data_len}.json', 'w'), ensure_ascii = False, indent = 2)

# enhanced_human_comment_list += enhanceData(human_comment_list[data_len:])
# json.dump(enhanced_human_comment_list, open('ExampleData/Comment/enhanced_human_comment_list.json', 'w'), ensure_ascii = False, indent = 2)

cleaned_human_list = []
comment_set = set()
for item in enhanced_human_comment_list:
    comment = item['comment']
    if 'rewrite_model' in item and comment not in comment_set:
        cleaned_human_list.append(item)
        comment_set.add(comment)
print(len(cleaned_human_list))
json.dump(cleaned_human_list, open('ExampleData/Comment/GPTEnhanced/enhanced_human_comment_list.json', 'w'), ensure_ascii = False, indent = 2)

# load old
# enhanced_GPT_comment_list = []
# with open('log/enhance_log', 'r') as f:
#     for line in f:
#         if line.startswith("{'comment':"):
#             if line.find('"') != -1:
#                 data = json.loads(line.strip().replace("\"", "$").replace("'", "\""))
#                 data['comment'] = data['comment'].replace("$", "\"")
#             else:
#                 data = json.loads(line.strip().replace("'", "\""))
#             if data['label'] == 'gpt-3.5-turbo':
#                 enhanced_GPT_comment_list.append(data)
# data_len = len(enhanced_GPT_comment_list)
# json.dump(enhanced_GPT_comment_list, open(f'ExampleData/Comment/enhanced_GPT_comment_list_{data_len}.json', 'w'), ensure_ascii = False, indent = 2)

# enhanced_GPT_comment_list = enhanceData(GPT_comment_list[data_len:])
# json.dump(enhanced_GPT_comment_list, open('ExampleData/Comment/enhanced_GPT_comment_list.json', 'w'), ensure_ascii = False, indent = 2)


# load old
# enhanced_GPT4_comment_list = []
# with open('log/enhance_log', 'r') as f:
#     for line in f:
#         if line.startswith("{'comment':"):
#             if line.find('"') != -1:
#                 data = json.loads(line.strip().replace("\"", "$").replace("'", "\""))
#                 data['comment'] = data['comment'].replace("$", "\"")
#             else:
#                 data = json.loads(line.strip().replace("'", "\""))
#             if data['label'] == 'gpt-4':
#                 enhanced_GPT4_comment_list.append(data)
# data_len = len(enhanced_GPT4_comment_list)
# json.dump(enhanced_GPT4_comment_list, open(f'ExampleData/Comment/enhanced_GPT4_comment_list_{data_len}.json', 'w'), ensure_ascii = False, indent = 2)

# enhanced_GPT4_comment_list = enhanceData(GPT4_comment_list[data_len:])
# json.dump(enhanced_GPT4_comment_list, open('ExampleData/Comment/enhanced_GPT4_comment_list.json', 'w'), ensure_ascii = False, indent = 2)
# enhanced_GLM_comment_list = enhanceData(GLM_comment_list)
