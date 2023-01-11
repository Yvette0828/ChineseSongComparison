from DSSM import *
from PERT import *

class File:
    def __init__(self, threshold, DSSMtrain, DSSMtest, DSSMresult, PERTresult, pinyin_list, PERTresult_clean, mergeRsult):
        # global threshold
        # global DSSMtrain
        # global DSSMtest
        # global DSSMresult
        # global PERTresult
        # global pinyin_list
        # global PERTresult_clean
        # global mergeRsult

        threshold = 85
        DSSMtrain = "./DSSM/data/QA_DSP2_2020S2_2 (dw)_checked.xlsx"
        DSSMtest = "./DSSM/data/QA_DSP1_20221h_Suspense - DW.xlsx"
        DSSMresult = "./DSSM/output/2022_secondhalf_negative_threshold_0{threshold}.xlsx".format(threshold=threshold)
        PERTresult = "./PERT/Logs/Eval_Rslt_DWAVE_Q3.txt"
        pinyin_list = "./PERT/Corpus/PERT_title_pinyin_Q3.txt"
        PERTresult_clean = "./result/PERT_result.xlsx"
        mergeRsult = "./result/merge_result_0{threshold}.xlsx".format(threshold=threshold)
        
def main():
    print('model runing...')
	
if __name__ == '__main__':
    print('Start...')
    print('DSSM running...\n')
    from DSSM.test_bmat_contributors_match import *

    print('Data prcoessing...\n')
    from PERT.dssm_process import *

    print('PERT running...\n')
    from PERT.eval_title import *
        
    print('Data post processing...\n')
    from PERT.post_process import *

    print('End.')