# -*- coding: utf-8 -*-

# xpinyin 漢語拼音

# pip install xpinyin -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

from xpinyin import Pinyin
import pykakasi
import requests_oauthlib
import re
import pandas as pd
from dssm.model import DSSM
import nltk
import json
from opencc import OpenCC 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

threshold = 0.70
DSSMtrain = "./DSSM/data/QA_DSP2_2020S2_2 (dw)_checked.xlsx"
DSSMtest = "./DSSM/data/QA_DSP1_20221h_Suspense - DW.xlsx"

def char2chpinyin(inputLIST):
  p = Pinyin()
  pinyinLIST = []
  for i in inputLIST:
    resSTR = p.get_pinyin(i).lower() 
    splitSTR = resSTR.split('-')
    joinSTR = ' '.join(splitSTR)
    pinyinLIST.append(joinSTR)
  return pinyinLIST

# 簡轉繁

# pip install opencc-python-reimplemented

# pykakasi 日語拼音

# pip install pykakasi

def char2jppinyin(inputSTR):
  kks = pykakasi.kakasi()
  nameLIST = []
  for i in inputSTR:
    resLIST = kks.convert(i)
    cleanLIST =[]
    for item in resLIST:
        res = "{}".format(item['hepburn']).lower()
        cleanLIST.append(res)
    nameSTR = ' '.join(cleanLIST)
    nameLIST.append(nameSTR)

  return nameLIST

# 匹配流程

# 讀取並清理訓練資料

# import data
# train_file = "./DSSM/data/QA_DSP2_2020S2_2 (dw)_checked.xlsx"
train_file = DSSMtrain
print("train file: ", train_file)
df = pd.read_excel(train_file)
df.head(50)

# 篩選人工標記為 correct match 的資料
df_correct = df[df['Correct match (Y/N)'] == 'y']

len(df_correct)

# 取代 reported_contributors 中的 NaN
df_correct['reported_contributors_filled'] = df_correct['reported_contributors'].fillna(0)

# 清除 reported_contributors 名字間的空格
recon_space = []
for i in range(len(df_correct)):
  if df_correct['reported_contributors_filled'].iloc[i] != 0:
    name = re.sub('([\u3400-\u9fa5]+) ([\u3400-\u9fa5]+)', r'\1\2', df_correct['reported_contributors_filled'].iloc[i])
    recon_space.append(name)
  else:
    recon_space.append(df_correct['reported_contributors_filled'].iloc[i])

# 將清除空白格後的 reported_contributors 資料新增至 df_correct 中
df_correct['reported_contributors_clean'] = recon_space

# 去除 reported_contributors_clean 中的重複名字
clean_recon = []
for i in range(len(df_correct)):
  if df_correct['reported_contributors_clean'].iloc[i] != 0:
    data = df_correct['reported_contributors_clean'].iloc[i].split('|')
    res = list(set(data))
    clean_recon.append(res)
  else:
    clean_recon.append([])

clean_recon

# 取代 matched_contributors 中的 NaN
df_correct['matched_contributors_filled'] = df_correct['matched_contributors'].fillna(0)

# 清除 matched_contributors 名字間的空格
macon_space = []
for i in range(len(df_correct)):
  if df_correct['matched_contributors_filled'].iloc[i] != 0:
    name = re.sub('([\u3400-\u9fa5]+) ([\u3400-\u9fa5]+)', r'\1\2', df_correct['matched_contributors_filled'].iloc[i])
    macon_space.append(name)
  else:
    macon_space.append(df_correct['matched_contributors_filled'].iloc[i])

# 將清除空白格後的 matched_contributors 資料新增至 df_correct 中
df_correct['matched_contributors_clean'] = macon_space

# 去除 matched_contributors_clean 中的重複名字
clean_macon = []
for i in range(len(df_correct)):
  if df_correct['matched_contributors_clean'].iloc[i] != 0:
    data = df_correct['matched_contributors_clean'].iloc[i].split('|')
    res = list(set(data))
    clean_macon.append(res)
  else:
    clean_macon.append([])

# 轉換漢語拼音

pin_recon_data = []
for i in clean_recon:
  res = char2chpinyin(i)
  pin_recon_data.append(res)

pin_recon_data

pin_macon_data = []
for i in clean_macon:
  res = char2chpinyin(i)
  pin_macon_data.append(res)

pin_macon_data

# 轉換拼音後，再次刪去重複者
pin_recon_nondup = []
for i in pin_recon_data:
  data = list(set(i))
  pin_recon_nondup.append(data)

pin_macon_nondup = []
for i in pin_macon_data:
  data = list(set(i))
  pin_macon_nondup.append(data)

# 訓練模型

# 準備訓練資料
split_queries = []
for i in pin_recon_nondup:
  result = ', '.join(i)
  split_queries.append(result)

print(split_queries)

# 準備訓練資料
split_documents = []
for i in pin_macon_nondup:
  result = ', '.join(i)
  split_documents.append(result)

print(split_documents)

# pip install dssm
nltk.download('punkt')

model = DSSM('dssm-model', device='cuda:0', lang='en')
model.fit(split_queries, split_documents)

# 模型測試

test_file = DSSMtest
print("test file: ", test_file)
df = pd.read_excel(test_file)
df.head(50)

# 選取出國語歌曲
df_ch = df[(df['reported_language']=='國語歌曲')|(df['owner']=='Ting')]

len(df_ch)

# 篩掉人工標記為 unclear 或 unknown 的資料
df_ch_nounclear = df_ch[(df_ch['inspection result']!='unclear') & (df_ch['inspection result']!='unknown')]

len(df_ch_nounclear)

df_ch_nounclear.head()

# 清理 reported contributors 資料

# 取代 reported_contributors 中的 NaN
df_ch_nounclear['reported_contributors_filled'] = df_ch_nounclear['reported_contributors'].fillna(0)

# 清除名字間的空格
recon_space = []
for i in range(len(df_ch_nounclear)):
  if df_ch_nounclear['reported_contributors_filled'].iloc[i] != 0:
    name = re.sub('([\u3400-\u9fa5]+) ([\u3400-\u9fa5]+)', r'\1\2', df_ch_nounclear['reported_contributors_filled'].iloc[i])
    recon_space.append(name)
  else:
    recon_space.append(df_ch_nounclear['reported_contributors_filled'].iloc[i])

# 將清除空白格的資料新增至 df_ch_nounclear
df_ch_nounclear['reported_contributors_clean'] = recon_space

# 去除重複名字
clean_recon = []
for i in range(len(df_ch_nounclear)):
  if df_ch_nounclear['reported_contributors_clean'].iloc[i] != 0:
    data = df_ch_nounclear['reported_contributors_clean'].iloc[i].split('|')
    res = list(set(data))
    clean_recon.append(res)
  else:
    clean_recon.append([])

# 清理 matched contributors 資料

# 取代 matched_contributors 中的 NaN
df_ch_nounclear['matched_contributors_filled'] = df_ch_nounclear['matched_contributors'].fillna(0)

# 清除名字間的空格
macon_space = []
for i in range(len(df_ch_nounclear)):
  if df_ch_nounclear['matched_contributors_filled'].iloc[i] != 0:
    name = re.sub('([\u3400-\u9fa5]+) ([\u3400-\u9fa5]+)', r'\1\2', df_ch_nounclear['matched_contributors_filled'].iloc[i])
    macon_space.append(name)
  else:
    macon_space.append(df_ch_nounclear['matched_contributors_filled'].iloc[i])

# 將清除空白格的資料新增至 df_ch_nounclear
df_ch_nounclear['matched_contributors_clean'] = macon_space

# 去除重複名字
clean_macon = []
for i in range(len(df_ch_nounclear)):
  if df_ch_nounclear['matched_contributors_clean'].iloc[i] != 0:
    data = df_ch_nounclear['matched_contributors_clean'].iloc[i].split('|')
    res = list(set(data))
    clean_macon.append(res)
  else:
    clean_macon.append([])

# 對照作詞作曲人字典並轉換

def con2dict(inputLIST):
  with open('./DSSM/data/contributors_dict.json') as f:
    conDICT = json.load(f)
  nameLIST = []
  cc = OpenCC('s2t')
  for i in inputLIST:
    s2tSTR = cc.convert(i).lower()  
    key = [k for k, v in conDICT.items() if any(s2tSTR in elem for elem in v)]
    if key != []:
      nameLIST.extend(key)
    else:
      nameLIST.append(s2tSTR)
      
  return nameLIST


clean_recon_dict = []
for i in clean_recon:
  clean_recon_dict.append(con2dict(i))

clean_macon_dict = []
for i in clean_macon:
  clean_macon_dict.append(con2dict(i))

# 轉換漢語拼音

chpin_recon_data = []
for i in clean_recon_dict:
  res = char2chpinyin(i)
  chpin_recon_data.append(res)

chpin_recon_data

chpin_macon_data = []
for i in clean_macon_dict:
  res = char2chpinyin(i)
  chpin_macon_data.append(res)

chpin_macon_data

# 轉換拼音後，再次刪去重複者
chpin_recon_nondup = []
for i in chpin_recon_data:
  data = list(set(i))
  chpin_recon_nondup.append(data)

chpin_macon_nondup = []
for i in chpin_macon_data:
  data = list(set(i))
  chpin_macon_nondup.append(data)

# 計算完全匹配之作詞作曲人個數
chsim_count = []
chsame_con = []
for i in range(len(df_ch_nounclear)):
  count = 0 
  res = list(set(chpin_recon_nondup[i]) & set(chpin_macon_nondup[i]))
  chsame_con.append(res)
  count += len(res)
  chsim_count.append(count)

# 計算 similarity score

model = DSSM('dssm-model', device='cpu')
chmatch_score = []
for i in range(len(df_ch_nounclear)):
  vectors = model.encode([' '.join(chpin_recon_nondup[i]), ' '.join(chpin_macon_nondup[i])])
  score = cosine_similarity([vectors[0]], [vectors[1]])
  chmatch_score.append(str(score))
  print(score)

# 轉換日語拼音

jppin_recon_data = []
for i in clean_recon_dict:
  res = char2jppinyin(i)
  jppin_recon_data.append(res)

jppin_recon_data

jppin_macon_data = []
for i in clean_macon_dict:
  res = char2jppinyin(i)
  jppin_macon_data.append(res)

jppin_macon_data

# 轉換拼音後，再次刪去重複者
jppin_recon_nondup = []
for i in jppin_recon_data:
  data = list(set(i))
  jppin_recon_nondup.append(data)

jppin_macon_nondup = []
for i in jppin_macon_data:
  data = list(set(i))
  jppin_macon_nondup.append(data)

# 計算完全匹配之作詞作曲人個數
jpsim_count = []
jpsame_con = []
for i in range(len(df_ch_nounclear)):
  count = 0 
  res = list(set(jppin_recon_nondup[i]) & set(jppin_macon_nondup[i]))
  jpsame_con.append(res)
  count += len(res)
  jpsim_count.append(count)

# 計算 similarity score

model = DSSM('dssm-model', device='cpu')
jpmatch_score = []
for i in range(len(df_ch_nounclear)):
  vectors = model.encode([' '.join(jppin_recon_nondup[i]), ' '.join(jppin_macon_nondup[i])])
  score = cosine_similarity([vectors[0]], [vectors[1]])
  jpmatch_score.append(str(score))
  # print(score)

# 將漢語拼音和日語拼音的完全匹配個數相加
total_count = []
for i in range(len(df_ch_nounclear)):
  total_count.append(chsim_count[i] + jpsim_count[i])

# 篩選出 similarity score 最大者
max_score = []
for i in range(len(df_ch_nounclear)):
  max_score.append(max(chmatch_score[i], jpmatch_score[i]))

# 清除 similarity score 中的括號
similarity_score = []
for i in max_score:
  score = re.sub('\[\[(.+)\]\]', r'\1', i)
  similarity_score.append(score)

# 將匹配結果製作成表格
pd.set_option('display.max_rows', 3000)
df_match = pd.DataFrame(zip(clean_recon_dict, clean_macon_dict, similarity_score, total_count), columns =['reported contributors', 'matched contributors', 'similarity score', 'same counts'])
# print(df_match)

# 結果評估

# threshold = 0.85
test_res = []
for i in range(len(df_match)):
  if df_match['reported contributors'][i] == []:
    test_res.append('correct')
  elif df_match['same counts'][i] != 0 and float(df_match['similarity score'][i]) > threshold:
    test_res.append('correct')
  elif df_match['same counts'][i] != 0:
    test_res.append('correct')
  elif float(df_match['similarity score'][i]) > threshold:
    test_res.append('correct')
  else:
    test_res.append('incorrect')

df_compare = pd.DataFrame(zip(df_ch_nounclear['dsp_song_id'], df_ch_nounclear['matched_work_id'], df_ch_nounclear['reported_title'], df_ch_nounclear['matched_mwtitle'], clean_recon_dict, clean_macon_dict, similarity_score, total_count, test_res, df_ch_nounclear['inspection result']), columns =['dsp_song_id', 'matched_work_id', 'reported title', 'matched title', 'reported contributors', 'matched contributors', 'similarity score', 'same counts', 'result', 'gold label'])
# df_compare = pd.DataFrame(zip(df_ch_nounclear['reported_title'], df_ch_nounclear['matched_mwtitle'], clean_recon_dict, clean_macon_dict, similarity_score, total_count, test_res, df_ch_nounclear['inspection result']), columns =['reported title', 'matched title', 'reported contributors', 'matched contributors', 'similarity score', 'same counts', 'result', 'gold label'])
print(df_compare.head(10))

correct = 0
for i in range(len(df_compare)):
  if df_compare['result'][i] == df_compare['gold label'][i]:
    correct += 1
  else:
    correct += 0

# accuracy
correct/len(df_compare)

# df_value = df_compare.iloc[:, 6:8]
# print(df_value.head())
# print(df_value.columns.to_list())
# get: ['similarity score', 'same counts']
df_value = df_compare
# map_ans = {'correct':1, 'incorrect':2}
# df_value.insert(2, 'gold label', 0)
# df_value['gold label'] = df_value['gold label'].map(map_ans)
# df_value['result'] = df_value['result'].map(map_ans)

y_true = df_value['gold label']
y_pred = df_value['result']  

print('macro:', f1_score(y_true, y_pred, average='macro'))
print('micro:', f1_score(y_true, y_pred, average='micro'))
print('weighted:', f1_score(y_true, y_pred, average='weighted'))

# perf_measure(y_true, y_pred)

threshold = str(threshold)

# False Positive
df_FP = df_compare[(df_compare['gold label']!='correct') & (df_compare['result']=='correct')]
print(df_FP)

FP_output_file = "./result/FP_threshold_0" + str(threshold[2:]) + ".xlsx"
# df_FP.to_excel(FP_output_file, encoding='utf-8')

# False Negative
df_FN = df_compare[(df_compare['gold label']!='incorrect') & (df_compare['result']=='incorrect')]
print(df_FN)

FN_output_file = "./result/FN_threshold_0" + str(threshold[2:]) + ".xlsx"
# df_FN.to_excel(FN_output_file, encoding='utf-8')

frames = [df_FP, df_FN]
df_false = pd.concat(frames)
df_false = df_false.rename(columns={'reported title':'reported_title', 'matched title':'matched_mwtitle'})

df_incorrect_match = pd.merge(df_ch_nounclear, df_false, on=['dsp_song_id', 'matched_work_id', 'reported_title', 'matched_mwtitle'], how='inner')
print(df_incorrect_match)

false_output_file = "./result/False_threshold_0" + str(threshold[2:]) + ".xlsx"
df_incorrect_match.to_excel(false_output_file, encoding='utf-8')