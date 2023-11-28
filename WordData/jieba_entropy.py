import json
import numpy as np

import jieba

# Metrics
# occur number for each words
word_dict = {}
unique_word_num = 0
# number of words in crawled comments
total_word_num = 0
# number of comments
comment_num = 0
# number of products
product_dict = {}
unique_product_num = 0

data_path = 'jingdong_comment_0907.json'
raw_data = json.load(open(data_path, 'r'))
raw_comment_list = raw_data['RECORDS']
print('Record number: ', len(raw_comment_list)) # 1015348

# Comment Format
# {
#   'commentText': '这个牌子的糯玉米?，买了几次了，个头大份量足，味道好有口感，一次蒸一根就可以了，搞活动优惠价买的，挺划算的，东北发货，几天才收到', 
#   'commentVariety': '玉米', 
#   'userStar': '5', 
#   'commentSenti': '好评', 
#   'createTime': '2022-01-01 00:00:26'
# }
for comment_item in raw_comment_list[:100000]:
    comment = comment_item['commentText']
    product = comment_item['commentVariety']
    # parsing words
    segment = list(jieba.cut(comment))
    for word in segment:
        word_dict[word] = word_dict.get(word, 0) + 1
    total_word_num += len(segment)
    product_dict[product] = product_dict.get(product, 0) + 1
    comment_num += 1
    if comment_num % 1000 == 0:
        print(f'comment_num: {comment_num}, word_dict: {len(word_dict)}')

unique_word_num = len(word_dict)
unique_product_num = len(product_dict)

# print(f'word_dict: {word_dict}')

statistic_dict = {
    'comment_num': comment_num,
    'unique_word_num': unique_word_num,
    'unique_product_num': unique_product_num,
    'total_word_num': total_word_num,
}
print('statistic_dict: ', statistic_dict)
print(f'product_dict: {product_dict}')

# dump result
import pickle
pickle.dump(word_dict, open('WordData/JiebaEntropy/word_dict.pkl', 'wb'))
pickle.dump(product_dict, open('WordData/JiebaEntropy/product_dict.pkl', 'wb'))
pickle.dump(statistic_dict, open('WordData/JiebaEntropy/statistic_dict.pkl', 'wb'))

# calculate entropy, sort by occurence
occur_word_list = [(word, word_dict[word], -np.log(word_dict[word] / total_word_num)) for word in word_dict]
occur_word_list = sorted(occur_word_list, key = lambda x: x[1], reverse = True)
print('Most occur words: ', occur_word_list[:10])
# save entropy
word_entropy_dict = {}
for word, occur, entropy in occur_word_list:
    word_entropy_dict[word] = entropy

# dump result
json.dump(word_entropy_dict, open('WordData/JiebaEntropy/word_entropy_dict.json', 'w'), ensure_ascii = False)
import pickle
pickle.dump(occur_word_list, open('WordData/JiebaEntropy/occur_word_list.pkl', 'wb'))
pickle.dump(word_entropy_dict, open('WordData/JiebaEntropy/word_entropy_dict.pkl', 'wb'))

# load datas
# import pickle
# word_dict = pickle.load(open('WordData/JiebaEntropy/word_dict.pkl', 'rb'))
# product_dict = pickle.load(open('WordData/JiebaEntropy/product_dict.pkl', 'rb'))
# statistic_dict = pickle.load(open('WordData/JiebaEntropy/statistic_dict.pkl', 'rb'))
# occur_word_list = pickle.load(open('WordData/JiebaEntropy/occur_word_list.pkl', 'rb'))
# word_entropy_dict = pickle.load(open('WordData/JiebaEntropy/word_entropy_dict.pkl', 'rb'))

# statistic log
# Record number:  1015348
# Building prefix dict from the default dictionary ...
# Loading model cost 0.399 seconds.
# Prefix dict has been built successfully.
# comment_num: 1000, word_dict: 3903
# comment_num: 2000, word_dict: 5619
# comment_num: 3000, word_dict: 6989
# comment_num: 4000, word_dict: 8192
# comment_num: 5000, word_dict: 9259
# comment_num: 6000, word_dict: 10225
# comment_num: 7000, word_dict: 11070
# comment_num: 8000, word_dict: 11871
# comment_num: 9000, word_dict: 12581
# comment_num: 10000, word_dict: 13312
# comment_num: 11000, word_dict: 13908
# comment_num: 12000, word_dict: 14693
# comment_num: 13000, word_dict: 15283
# comment_num: 14000, word_dict: 15899
# comment_num: 15000, word_dict: 16513
# comment_num: 16000, word_dict: 17118
# comment_num: 17000, word_dict: 17712
# comment_num: 18000, word_dict: 18268
# comment_num: 19000, word_dict: 18812
# comment_num: 20000, word_dict: 19394
# comment_num: 21000, word_dict: 19964
# comment_num: 22000, word_dict: 20436
# comment_num: 23000, word_dict: 20906
# comment_num: 24000, word_dict: 21361
# comment_num: 25000, word_dict: 21783
# comment_num: 26000, word_dict: 22224
# comment_num: 27000, word_dict: 22635
# comment_num: 28000, word_dict: 23085
# comment_num: 29000, word_dict: 23538
# comment_num: 30000, word_dict: 24033
# comment_num: 31000, word_dict: 24374
# comment_num: 32000, word_dict: 24750
# comment_num: 33000, word_dict: 25213
# comment_num: 34000, word_dict: 25724
# comment_num: 35000, word_dict: 26106
# comment_num: 36000, word_dict: 26480
# comment_num: 37000, word_dict: 26816
# comment_num: 38000, word_dict: 27131
# comment_num: 39000, word_dict: 27482
# comment_num: 40000, word_dict: 27804
# comment_num: 41000, word_dict: 28153
# comment_num: 42000, word_dict: 28519
# comment_num: 43000, word_dict: 28888
# comment_num: 44000, word_dict: 29234
# comment_num: 45000, word_dict: 29555
# comment_num: 46000, word_dict: 29887
# comment_num: 47000, word_dict: 30220
# comment_num: 48000, word_dict: 30529
# comment_num: 49000, word_dict: 30813
# comment_num: 50000, word_dict: 31128
# comment_num: 51000, word_dict: 31437
# comment_num: 52000, word_dict: 31720
# comment_num: 53000, word_dict: 31979
# comment_num: 54000, word_dict: 32230
# comment_num: 55000, word_dict: 32513
# comment_num: 56000, word_dict: 32803
# comment_num: 57000, word_dict: 33093
# comment_num: 58000, word_dict: 33323
# comment_num: 59000, word_dict: 33622
# comment_num: 60000, word_dict: 33882
# comment_num: 61000, word_dict: 34193
# comment_num: 62000, word_dict: 34449
# comment_num: 63000, word_dict: 34682
# comment_num: 64000, word_dict: 35000
# comment_num: 65000, word_dict: 35257
# comment_num: 66000, word_dict: 35486
# comment_num: 67000, word_dict: 35769
# comment_num: 68000, word_dict: 35979
# comment_num: 69000, word_dict: 36225
# comment_num: 70000, word_dict: 36478
# comment_num: 71000, word_dict: 36736
# comment_num: 72000, word_dict: 37015
# comment_num: 73000, word_dict: 37286
# comment_num: 74000, word_dict: 37549
# comment_num: 75000, word_dict: 37786
# comment_num: 76000, word_dict: 38046
# comment_num: 77000, word_dict: 38282
# comment_num: 78000, word_dict: 38497
# comment_num: 79000, word_dict: 38745
# comment_num: 80000, word_dict: 39014
# comment_num: 81000, word_dict: 39270
# comment_num: 82000, word_dict: 39532
# comment_num: 83000, word_dict: 39814
# comment_num: 84000, word_dict: 40049
# comment_num: 85000, word_dict: 40258
# comment_num: 86000, word_dict: 40496
# comment_num: 87000, word_dict: 40750
# comment_num: 88000, word_dict: 40959
# comment_num: 89000, word_dict: 41209
# comment_num: 90000, word_dict: 41437
# comment_num: 91000, word_dict: 41664
# comment_num: 92000, word_dict: 41904
# comment_num: 93000, word_dict: 42146
# comment_num: 94000, word_dict: 42395
# comment_num: 95000, word_dict: 42644
# comment_num: 96000, word_dict: 42882
# comment_num: 97000, word_dict: 43109
# comment_num: 98000, word_dict: 43365
# comment_num: 99000, word_dict: 43586
# comment_num: 100000, word_dict: 43832
# statistic_dict:  {'comment_num': 100000, 'unique_word_num': 43832, 'unique_product_num': 29, 'total_word_num': 2448447}
# product_dict: {'玉米': 6503, '棉花': 5185, '花生': 6685, '小麦': 7421, '枇杷': 1360, '蜜柚': 3618, '姜': 7622, '洋芋': 1554, '茶叶': 12555, '大米': 13138, '大蒜': 3697, '大豆': 2455, '芝麻': 6167, '荸荠': 275, '菊花': 1743, '竹荪': 1559, '藕': 1784, '豆皮': 2063, '辣椒': 1764, '椪柑': 386, '竹笋': 2262, '梨': 2752, '番茄': 2923, '猕猴桃': 3029, '李子': 329, '蜜茄': 232, '茶油': 844, '脚板薯': 28, '橙皮': 67}
# Most occur words:  [('，', 303154, 2.0889682977836914), ('的', 102240, 3.175886234373215), ('。', 71714, 3.530523238437705), ('很', 61014, 3.6921058789653842), ('了', 57877, 3.7448891559577078), ('好', 39550, 4.125643530788777), ('！', 32035, 4.336380169911455), ('不错', 27301, 4.496285893445759), (' ', 24937, 4.586856580678677), ('也', 24795, 4.592567205061342)]
