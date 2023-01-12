from cmath import nan
import re
import pandas as pd
from PERT.dssm_process import threshold, DSSMresult, PERTresult, pinyin_file, chinese_file, PERTresult_clean, mergeRsult

# import data
file_path = PERTresult
df = pd.read_csv(file_path, header = None)
# print(df)
# print(df[0][0][3:])
# print(df[0][1])
# print(df[0][2])

### result to excel format ###
df.insert(1, 'Result', nan)
df.insert(2, 'Correct', nan)
df.insert(3, 'Total', nan)
df.insert(4, 'Precision', nan)
df.insert(5, 'Error Rate', nan)
df.columns = ['Golden', 'Result', 'Correct', 'Total', 'Precision', 'Error Rate']
# print(df)

for i in range(len(df)):
    if 'G:' in df.iloc[i, 0]:
        df.iloc[i, 0] = df.iloc[i, 0][3:]

for i in range(len(df)):
    if 'R:' in df.iloc[i, 0]:
        df.iloc[i-1, 1] = df.iloc[i, 0][3:]
        # df = df.drop(df.iloc[i, :], axis = 0)
# print(df)

str1 = 'Correct: '
str2 = 'Total: '
str3 = 'precision: '
str4 = 'error rate: '

for i in range(len(df)):
    try:
        if 'precision:' in df.iloc[i, 0]:
            # print(df.iloc[i, 0])
            result_txt = df.iloc[i, 0]
            df.loc[i-2, 'Correct'] = result_txt[:df.iloc[i, 0].index(str2)]
            df.loc[i-2, 'Total'] = result_txt[df.iloc[i, 0].index(str2):df.iloc[i, 0].index(str3)]
            df.loc[i-2, 'Precision'] = result_txt[df.iloc[i, 0].index(str3):df.iloc[i, 0].index(str4)]
            df.loc[i-2, 'Error Rate'] = result_txt[df.iloc[i, 0].index(str4):]

    except:
        pass

        # df.iloc[i-2, 2] = df.iloc[i, 0]
        # df = df.drop(df.iloc[i, :], axis = 0)

result = df.dropna(axis = 0, how = 'any')
result = result.reset_index(drop = True)
print(result)

# for i in range(len(result)):
#     result.iloc[i, 2] = result.iloc[i, 2][48:]

pinyin_list = pd.read_csv(pinyin_file, sep="\t",header=None)
pinyin_list = pinyin_list.values.tolist()
# print(pinyin_list[0])

# try:
#     result.insert(0, 'PinYin', pinyin_list)
# except KeyError:
#     print("numbers of pinyin title are not match with Chinese title's")

for i in range(len(result)):
    # for specialChar in specialChars:
    # pinyin = str(result.iloc[i, 0])
    # print(txt[-1])
    # pinyin = pinyin.replace(pinyin[0], "") # replace [
    # pinyin = pinyin.replace(pinyin[1], "") # replace '
    # pinyin = pinyin.replace(pinyin[-2], "") # replace'
    # pinyin = pinyin.replace(pinyin[-1], "") # replace ]
    # result.iloc[i, 0] = pinyin

    correct = str(result.loc[i, 'Correct'])
    correct = correct.replace('Correct: ', "")
    result.loc[i, 'Correct'] = correct

    total = str(result.loc[i, 'Total'])
    total = total.replace('Total: ', "")
    result.loc[i, 'Total'] = total    

    precision = str(result.loc[i, 'Precision'])
    precision = precision.replace('precision: ', "")
    result.loc[i, 'Precision'] = precision

    error = str(result.loc[i, 'Error Rate'])
    error = error.replace('error rate: ', "")
    result.loc[i, 'Error Rate'] = error 

result['Correct'] = result['Correct'].astype(int)
result['Total'] = result['Total'].astype(int)
result['Precision'] = result['Precision'].astype(float)
result['Error Rate'] = result['Error Rate'].astype(float)

# print(result)

## Build Threshold Table ##
total_data = len(pinyin_list)
# print(total_data)
get_point = 0

threshold1= pd.DataFrame(columns = ['precision', '#count', 'correct_ratio(%)'], index = ['1-3', '4-6', '7-9'])
threshold1['precision'] = ['>=0.6', '>=0.6', '>=0.7']

for i in range(len(result)):
    if result.loc[i, 'Total']<4 and result.loc[i, 'Precision'] >=0.6:
        get_point = get_point + 1
        threshold1.iloc[0, 1] = get_point
    elif 3<result.loc[i, 'Total']<7 and result.loc[i, 'Precision'] >=0.6:
        get_point = get_point + 1
        threshold1.iloc[1, 1] = get_point
    elif 6<result.loc[i, 'Total']<10 and result.loc[i, 'Precision'] >=0.7:
        get_point = get_point + 1
        threshold1.iloc[2, 1] = get_point

for i in range(len(threshold1)):
    threshold1.iloc[i, 2] = round((threshold1.iloc[i, 1]/total_data)*100, 2)

# threshold1. = round((get_point / total_data)*100, 2)

# print(get_point)
print(threshold1)

# threshold = [1, 0.8, 0.7, 0.6, 0.5]
# count_1 = 0
# count_08 = 0
# count_07 = 0
# count_06 = 0
# count_05 = 0
# eva = pd.DataFrame(columns=['Count', 'Correct Ratio (%)'], index = threshold)

# for i in range(len(result)):
#     if result.loc[i, 'Precision'] == 1:
#         count_1 = count_1 + 1
#     eva.iloc[0,0] = count_1
#     eva.iloc[0,1] = round(count_1/total_data, 2)*100

#     if result.loc[i, 'Precision'] >= 0.8:
#         count_08 = count_08 + 1
#     eva.iloc[1,0] = count_08
#     eva.iloc[1,1] = round(count_08/total_data, 2)*100

#     if result.loc[i, 'Precision'] >= 0.7:
#         count_07 = count_07 + 1
#     eva.iloc[2,0] = count_07
#     eva.iloc[2,1] = round(count_07/total_data, 2)*100

#     if result.loc[i, 'Precision'] >= 0.6:
#         count_06 = count_06 + 1
#     eva.iloc[3,0] = count_06
#     eva.iloc[3,1] = round(count_06/total_data, 2)*100

#     if result.loc[i, 'Precision'] >= 0.5:
#         count_05 = count_05 + 1
#     eva.iloc[4,0] = count_05
#     eva.iloc[4,1] = round(count_05/total_data, 2)*100

# print(eva)

# result.to_excel('./Logs_excel/result_20221123.xlsx')

path = PERTresult_clean
writer = pd.ExcelWriter(path)
result.to_excel(writer, sheet_name='sheet1')
threshold1.to_excel(writer, sheet_name='threshold1')
# threshold2.to_excel(writer, sheet_name='threshold2')
writer.save()

#  merge PERT and DSSM result
dssm_file = DSSMresult
dssm = pd.read_excel(dssm_file)
print(dssm)

clean_file = "./PERT/Corpus/clean_test_id.xlsx"
clean = pd.read_excel(clean_file)
print(clean)

output = pd.concat([clean, result], axis=1)
output2 = pd.merge(dssm, output, on="dsp_song_id", how="outer") 
print(output2)
output2.to_excel(mergeRsult)

# output["dsp_song_id"] = output["dsp_song_id"].astype(str)
# dssm["dsp_song_id"] = dssm["dsp_song_id"].astype(str)

# for i in range(len(output)):
#     if output.loc[i, "dsp_song_id"] in dssm["dsp_song_id"]:
#         print("yes")
#         print(dssm[dssm["dsp_song_id"] == output.loc[i, "dsp_song_id"]].index)