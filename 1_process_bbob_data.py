import os

import glob
import cocopp

import pandas as pd
import numpy as np
import sklearn.metrics

# ! the script will download the data automatically
# ! in case of problems you can manually download from https://numbbo.github.io/data-archive/bbob/ into the directory data/0_coco
# ! then assign None to ALG_LIST
# ALG_LIST = None
ALG_LIST = ["2022/CMA-ES-Akimoto_Gharafi.tgz",
            "2022/CMA-ES-pycma_Gharafi.tgz",
            "2022/DD-CMA-ES-Akimoto_Gharafi.tgz",
            "2022/DD-CMA-ES-pycma_Gharafi.tgz",
            "2018/CMAES-APOP-Var1_Nguyen.tgz",
            "2022/CMAES-APOP-MA_Nguyen.tgz",
            "2022/CMAES-APOP-KP_Nguyen.tgz",
            "2022/CMAES-APOP-KMA_Nguyen.tgz",
            "2020/HE-ES_Glasmachers.tgz",
            "2009/RANDOMSEARCH_auger_noiseless.tgz"
]

args = {"DIMENSION":10, #for this paper, we use the data for 10D only
        "PRECISIONS":[1e-8, 1e-5, 1e-3],
        "ALG_LIST":ALG_LIST}

def mk_dir(path):
    if not os.path.exists(path):
            os.mkdir(path)

def mk_alldir(path_list):
    for path in path_list:
        mk_dir(path)

def step1_process_cocofile(path_coco, alg_list=None):
    # cocopp: process coco files
    if alg_list is None: # in case you manually downloaded the data, pass alg_list as None
        alg_str = " ".join([alg for alg in glob.glob(f"{path_coco}/*.tgz")])
    else:
        for alg_name in ALG_LIST: # in case the script will download the data
            cocopp.archives.bbob.get(f"{alg_name}")
            alg_str = " ".join(ALG_LIST) 
    alg_data = cocopp.main(alg_str)

    # organize by algorithm and dimension
    data_coco = {}
    for key, data in alg_data.items():
        name = key[0].split('_')[0] #: shorten alg name by removing authors name
        data_coco[name] = data.dictByDim() #organize by DIM

    return data_coco

def step2_extract_metrics(data_coco, dimension, precisions):
     # extract the metrics of interest from the coco_objects and add them to a pandas dataframe
    def process_algdata(alg_data, dimension, precisions):
        assert alg_data.dim == dimension

        # alg name
        algname = alg_data.algId.split("_")[0] #: shorten alg name by removing authors name
        
        # function id
        fid = alg_data.funcId

        # indices for requested precisions
        ert_mn = []
        for p in precisions:
            try:
                idx = np.where(alg_data.target==p)[0][0]
                ert_mn += [alg_data.ert[idx]/dimension]
            except IndexError: 
                #the algorithm may not have solved that precision
                # leave it as nan
                ert_mn += [np.nan]

        # ert area
        x = np.log10(alg_data.ert)/dimension
        y = np.linspace(start=0, stop=1, num=len(x)) #0-100 scale
        ert_area = sklearn.metrics.auc(x=x,y=y)
        
        # avg error 
        errora = np.mean(alg_data.readfinalFminusFtarget)
        # error x<=1e-8 -> x=0
        errorb = np.mean([0 if v<=1e-8 else v for v in alg_data.readfinalFminusFtarget])
        
        return [algname, f"f{fid}d{dimension}", *ert_mn, ert_area, 
                errora, errorb]

    mycols = ["algorithm", "function", "ert_m8", "ert_m5", "ert_m3", "log10ert_area", 
            "error_avg", "errorbounded_avg"]

    df = pd.DataFrame([], columns=mycols)
    for alg in data_coco.keys():
        for f in range(0,24):
            row = process_algdata(data_coco[alg][dimension][f], dimension, precisions)
            df.loc[len(df.index)] = row

    return df

def main(args):
    # data path
    path = "./data"
    path_coco = f"{path}/0_coco"
    path_csv = f"{path}/1_csv"
    mk_alldir([path, path_coco, path_csv])

    # run cocopp to process coco data
    data_coco = step1_process_cocofile(path_coco, args["ALG_LIST"])
    # extract metrics of interest
    df = step2_extract_metrics(data_coco, args["DIMENSION"], args["PRECISIONS"])
    # save as csv
    df.to_csv(f"{path_csv}/bbob_fall_{args['DIMENSION']}d.csv", index=False)

if __name__ == "__main__":
    main(args)

