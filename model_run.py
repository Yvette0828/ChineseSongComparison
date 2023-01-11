from DSSM import *
from PERT import *
        
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