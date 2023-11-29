import json

# human data

data_path = ''
raw_data = json.load(open(data_path, 'r'))
raw_comment_list = raw_data['RECORDS'] # 1015348

human_comment_list = []
product_dict = {}
# Comment Format
# {
#   'commentText': '这个牌子的糯玉米?，买了几次了，个头大份量足，味道好有口感，一次蒸一根就可以了，搞活动优惠价买的，挺划算的，东北发货，几天才收到', 
#   'commentVariety': '玉米', 
#   'userStar': '5', 
#   'commentSenti': '好评', 
#   'createTime': '2022-01-01 00:00:26'
# }
for comment_item in raw_comment_list[:3000]:
    comment = comment_item['commentText']
    product = comment_item['commentVariety']
    human_comment_list.append({'comment': comment, 'product': product, 'label': 'human'})
    product_dict[product] = product_dict.get(product, 0) + 1

print('product_dict: ', product_dict)
# product_dict:  {'玉米': 214, '棉花': 142, '花生': 211, '小麦': 210, '枇杷': 17, '蜜柚': 156, '姜': 283, '洋芋': 55, '茶叶': 344, '大米': 494, '大蒜': 90, '大豆': 77, '芝麻': 145, '荸荠': 16, '菊花': 40, '竹荪': 35, '藕': 30, '豆皮': 49, '辣椒': 50, '椪柑': 24, '竹笋': 48, '梨': 56, '番茄': 76, '猕猴桃': 87, '李子': 10, '蜜茄': 7, '茶油': 32, '脚板薯': 2}
# json.dump(human_comment_list, open('ExampleData/Comment/human_comment_list.json', 'w'), ensure_ascii = False)


# GLM Data

import zhipuai
zhipuai.api_key = ''

'''
# Response Format
{
  "code": 200,
  "msg": "",
  "success": true,
  "data": {
      "task_id": "75931252186628016897601864755556524089",
      "request_id": "123445676789",
      "task_status": "SUCCESS",
      "choices": [
          {"role": "assistant", "content":"作为一个大型语言模型,我可以完成许多不同的任务,包括但不限于: \n1. 回答问题 \n2.提供建议……"}
      ],
      "usage": {
          "prompt_tokens": 215,
          "completion_tokens": 302,
          "total_tokens": 517
      }     
  }
}
'''
# GLM同步调用API
import zhipuai
def invoke_api(prompt = None, model = 'chatglm_turbo', top_p = 0.7, temperature = 0.9):
    assert prompt is not None
    response = zhipuai.model_api.invoke(
        model = model,
        prompt = prompt,
        top_p = top_p,
        temperature = temperature,
    )
    # print(response)
    response_text = response["data"]["choices"][0]["content"]
    return response_text

# 获取GLM生成的评论
def getCommentsWithGLM(product, temperature = 0.9):
    prompt = [{
        "role": "user", 
        "content": f'你是一名网购用户，在京东购买了{product}，请写一条购物评论，只写正文，不要任何标题。'
    }]
    response_text = invoke_api(prompt, temperature = temperature) # 同步调用
    response_text = response_text.replace('\n', '').replace(' ', '')
    print('GLM response:', response_text)
    return response_text

# getCommentsWithGLM('柑橘', temperature = 0.5)

# GLM examples
# temperature = 0.9
# 这次购买的柑橘实在是太满意了！首先，包装非常用心，每一个果实都得到了很好的保护。收到货时，尽管运输过程中有一些挤压，但大部分柑橘依然保持着新鲜的状态。其次，柑橘的口感和品质都非常优秀，果肉饱满，汁水丰富，酸甜适中，让人回味无穷。最后，京东的配送速度也值得点赞，当天下单，第二天就收到了，非常给力！\n\n总之，这次购物体验让我非常满意，不仅产品质量上乘，服务也到位。以后还会继续支持京东，也希望卖家能够继续保持品质，为大家带来更多优质的产品。
# 京东的柑橘真心不错，这次购物体验让我非常满意！首先，包装得很用心，确保了水果的新鲜度；其次，柑橘口感鲜美，汁多肉厚，吃起来特别清爽解渴；最后，价格实惠，让我觉得物超所值。希望京东能继续保持这种高品质，为广大消费者提供更多优质商品。总之，推荐给大家，值得购买！
# temperature = 0.99
# 京东的柑橘真心不错，这次购物体验让我非常满意！首先，包装得很用心，确保了果实的新鲜度；其次，柑橘的口感鲜美，果肉饱满，酸甜适中，非常好吃。此外，配送速度也很快，物流小哥态度很好。这次购买的柑橘品质上乘，价格实惠，让我感受到了京东商城的高品质服务。以后还会继续支持京东，期待更多优惠活动！
# temperature = 0.5
# 京东的柑橘真心不错，这次购物体验让我非常满意！首先，包装非常用心，确保了水果的新鲜度；其次，柑橘的口感鲜美，汁多肉厚，吃起来回味无穷；最后，京东的配送速度一如既往地给力，让我能在第一时间品尝到这美味的柑橘。总之，这次购物让我感受到了京东的用心和专业，以后我会继续支持京东，也希望京东能继续保持高品质的服务。总之，好评无疑！

# 写两条
# 这次的京东购物体验真的非常棒！我购买的柑橘品质优良，果实饱满，色泽鲜艳，口感清爽甘甜。物流速度也让人赞叹不已，包装完好无损，真心觉得物超所值。期待下次再次光顾，同时也推荐给朋友们哦！
# 在京东购买的柑橘让我惊艳不已！水果新鲜程度堪比现场采摘，味道浓郁，水分充足。包装也很用心，确保了果实的安全。这样的品质和服务，让我觉得购物无忧，给五星好评！希望商家继续保持，让更多人享受到这份美好。

# Web 10条
# 京东购物就是快，下单后隔天就收到了新鲜的柑橘，物流给力！味道酸甜适中，口感非常好。
# 这次购买的柑橘质量真的很棒，包装完好，果实饱满，京东自营就是靠谱！
# 家里的老人孩子都喜欢吃柑橘，这次买的口感鲜美，营养价值高，推荐给大家。
# 京东的售后服务真的很棒，柑橘使用过程中遇到问题，客服耐心解答，解决了我的疑虑。
# 这次购买的柑橘比预期的好，果实新鲜，颜色好看，非常适合全家享用，京东购物满意！
# 京东购物就是放心，柑橘质量有保证，价格还实惠，以后还会继续支持！
# 收到柑橘后立刻品尝了一下，水分充足，味道鲜美，真的是太好吃了，京东买的柑橘就是不一样！
# 这次购买的柑橘是给同事们带的，他们都表示口感很好，问我哪里买的，我会继续推荐的。
# 柑橘包装很好，适合送礼，自己吃也很划算，京东购物就是值得信赖！
# 京东柑橘品质稳定，每次购买都很满意，已经成为我固定购买的平台了！


# ChatGPT - Chatanywhere

import openai
openai.api_base = "https://api.chatanywhere.cn/v1"
openai.api_key = ''


# TODO: assistant指定多轮对话，如包含多轮对话，则msgs = [system, user, assistant, user]
def chatWithGPT(system, user, assistant = '', model = "gpt-3.5-turbo", temperature = 0.7):
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

def getCommentsWithGPT(product, temperature, model = 'gpt-3.5-turbo'):
    response = chatWithGPT(
        system = '你是一名网购用户',
        user = f'你在京东购买了{product}，请写一条购物评论，只写内容，不要有任何标题。',
        model = model,
        temperature = temperature,
    )
    response_text = response["choices"][0]["message"]["content"]
    return response_text

old_gpt_list = json.load(open('ExampleData/Comment/GPT_comment_list.json', 'r'))
json.dump(old_gpt_list, open('ExampleData/Comment/GPT_comment_list_old.json', 'w'), ensure_ascii = False)

GPT_comment_list = []
for product in product_dict:
    num = min(product_dict[product], 10)
    for idx in range(num):
        response_text = getCommentsWithGPT(product, 0.9)
        GPT_comment_list.append({'comment': response_text, 'product': product, 'label': 'gpt-3.5-turbo'})
json.dump(old_gpt_list + GPT_comment_list, open('ExampleData/Comment/GPT_comment_list.json', 'w'), ensure_ascii = False)


