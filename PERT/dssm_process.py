import re
from cmath import nan
import pandas as pd
import numpy as np
from opencc import OpenCC

global threshold
global DSSMresult
global PERTresult
global pinyin_file
global PERTresult_clean
global mergeRsult

threshold = 0.70
DSSMresult = "./result/False_threshold_0{threshold}.xlsx".format(threshold=str(threshold)[2:])
PERTresult = "./PERT/Logs/Eval_Rslt_DWAVE_test_0{threshold}.txt".format(threshold=str(threshold)[2:])
pinyin_file = "./PERT/Corpus/PERT_title_pinyin_test_0{threshold}.txt".format(threshold=str(threshold)[2:])
chinese_file = "./PERT/Corpus/PERT_title_Chinese_test_0{threshold}.txt".format(threshold=str(threshold)[2:])
PERTresult_clean = "./result/PERT_result_0{threshold}.xlsx".format(threshold=str(threshold)[2:])
mergeRsult = "./result/merge_result_0{threshold}.xlsx".format(threshold=str(threshold)[2:])

dssm_file = DSSMresult  # file name
print(dssm_file)
df = pd.read_excel(dssm_file)
df = df.astype(str)  # all column's dtype convert to "object"
# print(df)

df2 = df[['dsp_song_id', 'reported_title', 'matched_mwtitle']] #, 'gold label', 'enhanced_title', 'enhanced_mwtitle',
print(df2)

# 1 Filter out the song title with pinyin and Chinese
def is_contain_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def delete_non_chinese(strs):
  strs=list(strs)
  new_strs=[]
  for _char in strs:
    if '\u4e00' <= _char <= '\u9fa5':
      new_strs.append(_char)
  return ''.join(new_strs)

def delete_non_english(strs):
  strs=list(strs)
  new_strs=[]
  for _char in strs:
    if (_char >= u'\u0041' and _char <= u'\u005A') or (_char >= u'\u0061' and _char <= u'\u007A') or (_char == u'\u0020'):
      new_strs.append(_char)
  return ''.join(new_strs)
  # space \u0020

# 2 Remove punctuation and numbers
def delete_brackets(strs):
  strs=list(strs)
  new_strs=[]
  flag=0
  brackets={'(':')','《':'》'}
  left_brackets=''
  for _char in strs:
    if _char in brackets:
      flag=1
      left_brackets=_char
    if flag==0:
      new_strs.append(_char)
    if flag==1 and _char==brackets[left_brackets]:
      flag=0

  return ''.join(new_strs)

# 3 convert traditional to simplified Chinese
cc = OpenCC('t2s')

for i in df2.index:
  df2.loc[i,'reported_title']=delete_brackets(df2.loc[i,'reported_title'])
  df2.loc[i,'reported_title']=delete_non_chinese(df2.loc[i,'reported_title'])

  # 3 convert traditional to simplified Chinese
  df2.loc[i,'reported_title']=cc.convert(df2.loc[i,'reported_title'])

  df2.loc[i,'matched_mwtitle']=delete_brackets(df2.loc[i,'matched_mwtitle'])
  df2.loc[i,'matched_mwtitle']=delete_non_english(df2.loc[i,'matched_mwtitle'])

  # 4 convert all pinyin to lowercase
  df2.loc[i,'matched_mwtitle']=df2.loc[i,'matched_mwtitle'].lower()
  # df2.loc[i,'enhanced_mwtitle']=df2.loc[i,'enhanced_mwtitle'].lower()

  rep=df2.loc[i,'reported_title']
  mat=df2.loc[i,'matched_mwtitle']
  # enh=df2.loc[i,'enhanced_mwtitle']
  # if mat == 'nan'	and enh != 'nan':
  #   # print(mat)
  #   # print(enh)
  #   mat = enh
  rep_con = is_contain_chinese(rep)
  mat_con = is_contain_chinese(mat)
  if rep_con^mat_con == False:
      df2 = df2.drop([i])
print(df2)

# drop nan
df3 = df2[["dsp_song_id", "reported_title", "matched_mwtitle"]]
for i in range(len(df3)):
    if df3.iloc[i, 2] == '':
        df3.iloc[i, 2] = nan

df3 = df3.dropna(axis=0, how='any')
print(df3)

clean_file = "./PERT/Corpus/clean_test_id.xlsx"
df3.to_excel(clean_file)

# write the test file (Chinese)
with open(chinese_file, "w") as chinese_f:
    for i in range(len(df3)):
        chinese_title = df3.iloc[i, 1]
        # print(chinese_title)
        new_chinese_title = ' '.join(chinese_title)
        # print(new_chinese_title)
        chinese_f.write(new_chinese_title)
        chinese_f.write("\n")
chinese_f.close()

# write the test file (Pinyin)
with open(pinyin_file, "w") as pinyin_f:
    for i in range(len(df3)):
        pinyin_title = df3.iloc[i, 2]
        pinyin_f.write(pinyin_title)
        pinyin_f.write("\n")
pinyin_f.close()