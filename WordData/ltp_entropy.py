import json
import numpy as np

# LTP for parsing words
from ltp import LTP
ltp = LTP()

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
    segment = ltp.pipeline([comment], tasks=['cws']).cws[0]
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
pickle.dump(word_dict, open('WordData/LTPEntropy/word_dict.pkl', 'wb'))
pickle.dump(product_dict, open('WordData/LTPEntropy/product_dict.pkl', 'wb'))
pickle.dump(statistic_dict, open('WordData/LTPEntropy/statistic_dict.pkl', 'wb'))

# calculate entropy, sort by occurence
occur_word_list = [(word, word_dict[word], -np.log(word_dict[word] / total_word_num)) for word in word_dict]
occur_word_list = sorted(occur_word_list, key = lambda x: x[1], reverse = True)
# save entropy
word_entropy_dict = {}
for word, occur, entropy in occur_word_list:
    word_entropy_dict[word] = entropy

# dump result
json.dump(word_entropy_dict, open('WordData/LTPEntropy/word_entropy_dict.json', 'w'), ensure_ascii = False)
import pickle
pickle.dump(occur_word_list, open('WordData/LTPEntropy/occur_word_list.pkl', 'wb'))
pickle.dump(word_entropy_dict, open('WordData/LTPEntropy/word_entropy_dict.pkl', 'wb'))

# load datas
# import pickle
# word_dict = pickle.load(open('WordData/LTPEntropy/word_dict.pkl', 'rb'))
# product_dict = pickle.load(open('WordData/LTPEntropy/product_dict.pkl', 'rb'))
# statistic_dict = pickle.load(open('WordData/LTPEntropy/statistic_dict.pkl', 'rb'))
# occur_word_list = pickle.load(open('WordData/LTPEntropy/occur_word_list.pkl', 'rb'))
# word_entropy_dict = pickle.load(open('WordData/LTPEntropy/word_entropy_dict.pkl', 'rb'))

# statistic log
# Record number:  1015348
# comment_num: 1000, word_dict: 3580
# comment_num: 2000, word_dict: 5177
# comment_num: 3000, word_dict: 6399
# comment_num: 4000, word_dict: 7461
# comment_num: 5000, word_dict: 8413
# comment_num: 6000, word_dict: 9302
# comment_num: 7000, word_dict: 10075
# comment_num: 8000, word_dict: 10819
# comment_num: 9000, word_dict: 11512
# comment_num: 10000, word_dict: 12140
# comment_num: 11000, word_dict: 12684
# comment_num: 12000, word_dict: 13453
# comment_num: 13000, word_dict: 13994
# comment_num: 14000, word_dict: 14541
# comment_num: 15000, word_dict: 15123
# comment_num: 16000, word_dict: 15709
# comment_num: 17000, word_dict: 16277
# comment_num: 18000, word_dict: 16800
# comment_num: 19000, word_dict: 17301
# comment_num: 20000, word_dict: 17859
# comment_num: 21000, word_dict: 18398
# comment_num: 22000, word_dict: 18888
# comment_num: 23000, word_dict: 19362
# comment_num: 24000, word_dict: 19792
# comment_num: 25000, word_dict: 20206
# comment_num: 26000, word_dict: 20666
# comment_num: 27000, word_dict: 21072
# comment_num: 28000, word_dict: 21531
# comment_num: 29000, word_dict: 21958
# comment_num: 30000, word_dict: 22452
# comment_num: 31000, word_dict: 22790
# comment_num: 32000, word_dict: 23197
# comment_num: 33000, word_dict: 23671
# comment_num: 34000, word_dict: 24140
# comment_num: 35000, word_dict: 24492
# comment_num: 36000, word_dict: 24869
# comment_num: 37000, word_dict: 25232
# comment_num: 38000, word_dict: 25536
# comment_num: 39000, word_dict: 25864
# comment_num: 40000, word_dict: 26198
# comment_num: 41000, word_dict: 26504
# comment_num: 42000, word_dict: 26886
# comment_num: 43000, word_dict: 27283
# comment_num: 44000, word_dict: 27652
# comment_num: 45000, word_dict: 27983
# comment_num: 46000, word_dict: 28349
# comment_num: 47000, word_dict: 28663
# comment_num: 48000, word_dict: 28976
# comment_num: 49000, word_dict: 29250
# comment_num: 50000, word_dict: 29547
# comment_num: 51000, word_dict: 29841
# comment_num: 52000, word_dict: 30184
# comment_num: 53000, word_dict: 30430
# comment_num: 54000, word_dict: 30715
# comment_num: 55000, word_dict: 31001
# comment_num: 56000, word_dict: 31282
# comment_num: 57000, word_dict: 31550
# comment_num: 58000, word_dict: 31823
# comment_num: 59000, word_dict: 32119
# comment_num: 60000, word_dict: 32410
# comment_num: 61000, word_dict: 32721
# comment_num: 62000, word_dict: 32991
# comment_num: 63000, word_dict: 33242
# comment_num: 64000, word_dict: 33565
# comment_num: 65000, word_dict: 33837
# comment_num: 66000, word_dict: 34075
# comment_num: 67000, word_dict: 34358
# comment_num: 68000, word_dict: 34582
# comment_num: 69000, word_dict: 34858
# comment_num: 70000, word_dict: 35127
# comment_num: 71000, word_dict: 35428
# comment_num: 72000, word_dict: 35718
# comment_num: 73000, word_dict: 35988
# comment_num: 74000, word_dict: 36245
# comment_num: 75000, word_dict: 36483
# comment_num: 76000, word_dict: 36731
# comment_num: 77000, word_dict: 36971
# comment_num: 78000, word_dict: 37230
# comment_num: 79000, word_dict: 37517
# comment_num: 80000, word_dict: 37779
# comment_num: 81000, word_dict: 38038
# comment_num: 82000, word_dict: 38303
# comment_num: 83000, word_dict: 38561
# comment_num: 84000, word_dict: 38821
# comment_num: 85000, word_dict: 39032
# comment_num: 86000, word_dict: 39284
# comment_num: 87000, word_dict: 39514
# comment_num: 88000, word_dict: 39723
# comment_num: 89000, word_dict: 39988
# comment_num: 90000, word_dict: 40239
# comment_num: 91000, word_dict: 40484
# comment_num: 92000, word_dict: 40739
# comment_num: 93000, word_dict: 40996
# comment_num: 94000, word_dict: 41263
# comment_num: 95000, word_dict: 41552
# comment_num: 96000, word_dict: 41810
# comment_num: 97000, word_dict: 42063
# comment_num: 98000, word_dict: 42304
# comment_num: 99000, word_dict: 42546
# comment_num: 100000, word_dict: 42791
# statistic_dict:  {'comment_num': 100000, 'unique_word_num': 42791, 'unique_product_num': 29, 'total_word_num': 2592948}
# product_dict: {'玉米': 6503, '棉花': 5185, '花生': 6685, '小麦': 7421, '枇杷': 1360, '蜜柚': 3618, '姜': 7622, '洋芋': 1554, '茶叶': 12555, '大米': 13138, '大蒜': 3697, '大豆': 2455, '芝麻': 6167, '荸荠': 275, '菊花': 1743, '竹荪': 1559, '藕': 1784, '豆皮': 2063, '辣椒': 1764, '椪柑': 386, '竹笋': 2262, '梨': 2752, '番茄': 2923, '猕猴桃': 3029, '李子': 329, '蜜茄': 232, '茶油': 844, '脚板薯': 28, '橙皮': 67}

