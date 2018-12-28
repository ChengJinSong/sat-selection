# pylint: disable=C0103,C0111
import sys
import pandas as pd

# ALGO_RUN = "D:/ASLIB/SAT12-INDU/algorithm_runs.arff"
# CSV_NAME = 'D:/SAT-INSTANCE/SAT12-indu.csv'


def get_csv(algo_run, csv_name):
    f = open(algo_run)
    print(f.readline())
    lines = f.readlines()
    isdata = 0
    dic = {'filename': [], 'repetition': [], 'algorithm': [], 'runtime': [], 'runstatus': []}
    for line in lines:
        if not isdata:
            if line == '@DATA\n':
                isdata = 1
            continue
        filename, repetition, algorithm, runtime, runstatus = line.split(',')
        dic['filename'].append(filename)
        dic['repetition'].append(repetition)
        dic['algorithm'].append(algorithm)
        dic['runtime'].append(runtime)
        dic['runstatus'].append(runstatus.rstrip())
    f.close()
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(csv_name, index=False)


# if len(sys.argv) > 1:
#     data_name = sys.argv[1]
#     ALGO_RUN = 'D:/ASLIB/{}/algorithm-run.arff'.format(data_name)
#     CSV_NAME = 'D:/SAT-INSTANCE/{}.csv'.format(data_name)
#     get_csv(ALGO_RUN, CSV_NAME)
# else:
#     print('Invalid data name')
