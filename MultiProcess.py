import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import multiprocessing
import gc
from datetime import datetime

def main(file, start, end, result, lock):
    SUGGESTS = file['SUGGEST_DUMMY'].unique()[start: end]
    f = pd.DataFrame()
    names = pd.get_dummies(file['NAMES'])
    categories = pd.get_dummies(file['CATEGORIES'])
    types = pd.get_dummies(file['TYPE'])
    for idx, obj in enumerate(SUGGESTS):
        cond = file['SUGGEST_DUMMY'] == obj
        TEMP = file[cond].reset_index(drop = True)
        if len(TEMP['BP_DUMMY'].unique()) > 1:
            continue
        temp = pd.DataFrame(np.zeros(shape = (1, 15)),columns = ['DATE', 'AGENT_DUMMY',
                                                                'SUGGEST_DUMMY', 'SUGGEST_DATE',
                                                                'SUGGEST_LAST_DATE', 'AGE',
                                                                'BP_DUMMY', 'AMOUNT', 'CURRENCY',
                                                                'FEE', 'ACCEPT',
                                                                'FORTUNELV', 'LOYALTYLV', 'CLUSTER',
                                                                 'NEW_OLD'])
        temp['DATE'] = TEMP['DATE'].iloc[0]
        temp['SUGGEST_DUMMY'] = obj
        temp['AGENT_DUMMY'] = TEMP['AGENT_DUMMY'].iloc[-1]
        temp['SUGGEST_DATE'] = TEMP['SUGGEST_DATE'].iloc[0]
        temp['SUGGEST_LAST_DATE'] = TEMP['SUGGEST_LAST_DATE'].iloc[-1]
        try:
            temp['AGE'] = TEMP['AGE'][TEMP['AGE'].notna()].iloc[-1]
        except:
            temp['AGE'] = np.nan
        try:
            temp['FEE'] = np.sum(TEMP['FEE'][TEMP['FEE'].notna()])
        except:
            temp['FEE'] = np.nan
        try:
            temp['AMOUNT'] = np.sum(TEMP['AMOUNT'][TEMP['AMOUNT'].notna()])
        except:
            temp['AMOUNT'] = np.nan

        temp['BP_DUMMY'] = TEMP['BP_DUMMY'].iloc[-1]
        # temp['AMOUNT'] = np.sum(TEMP['AMOUNT'][TEMP['AMOUNT'].notna()])
        # temp['FEE'] = np.sum(TEMP['FEE'][TEMP['FEE'].notna()])
        temp['ACCEPT'] = TEMP['ACCEPT'].iloc[-1]
        temp['FORTUNELV'] = TEMP['FORTUNELV'].iloc[-1]
        temp['LOYALTYLV'] = TEMP['LOYALTYLV'].iloc[-1]
        temp['CLUSTER'] = TEMP['CLUSTER'].iloc[-1]
        temp['NEW_OLD'] = (1 if temp['CLUSTER'].isna().any() else 0)
        A = (pd.DataFrame(np.sum(categories[cond], axis = 0)) > 0).astype(int).T
        B = (pd.DataFrame(np.sum(names[cond], axis = 0)) > 0).astype(int).T
        C = (pd.DataFrame(np.sum(types[cond], axis = 0)) > 0).astype(int).T
        temp = pd.concat([temp, A, B, C], axis = 1)
        f = pd.concat([f, temp], axis = 0)
        del temp, TEMP, cond, A, B, C
        gc.collect()
    lock.acquire()
    result.append(f)
    lock.release()
       
if __name__=='__main__':
    # Data Reading
    warnings.filterwarnings('ignore')
    address = '/home/309707001/thesis/comb/'
    File = '/home/309707001/thesis/comb/CombineData.csv'
    f = pd.read_csv(File, encoding = 'Big5').iloc[:,1:]
    F = pd.DataFrame()
    F['SUGGEST_DATE'] = f['SUGGEST_DATE'].apply(lambda x: datetime.strptime(x, "%Y/%m/%d"))
    I = np.argsort(F['SUGGEST_DATE'])
    f = f.iloc[I]
    # Processing
    lock = multiprocessing .Lock()
    processes = list()
    process_amount = 60
    result = multiprocessing.Manager().list()
    lapse = len(f['SUGGEST_DUMMY'].unique())//process_amount
    for idx in range(process_amount):
        if idx != process_amount - 1:
            start = idx*lapse
            end = lapse*(idx + 1)
        else:
            start = lapse*idx
            end = f.shape[0]
        processes.append(multiprocessing.Process(target = main, args = (f, start, end, result, lock)))
        processes[idx].start()
    print('start running')
    for idx in range(process_amount):
        processes[idx].join()
    RESULT = pd.DataFrame()
    for obj in result:
        RESULT = pd.concat([RESULT, obj], axis = 0)
    print(result)
    RESULT.to_csv(address + 'Intermediate.csv', encoding = 'Big5')

