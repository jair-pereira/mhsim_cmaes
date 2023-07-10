import os
import itertools
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from scipy import stats

args = {"MYORDER": [#algorithm order in the figures
                    'CMA-ES-Akimoto', 
                    'CMA-ES-pycma',
                    'DD-CMA-ES-Akimoto',
                    'DD-CMA-ES-pycma',
                    'CMAES-APOP-Var1',
                    'CMAES-APOP-KMA',
                    'CMAES-APOP-KP',
                    'CMAES-APOP-MA',
                    'HE-ES',
                    'RANDOMSEARCH'],

        "DIMENSIONS": [10], # problems dimension of interest
        "METRICS"   : ["log10ert_area"], # metrics of interest

        "PATH_COMP": "data/",
        "PATH_CSV": "data/1_csv",
        "PATH_FIG" : "data/2_figures",
        "PATH_CORR" : "data/3_pearson"
}

def calc_component_similarity(a, b):
    mask = np.logical_and((a!="X"), (b!="X"))

    num_components   = sum(a!="X")
    num_U_components = sum(a[mask]==b[mask])

    return num_U_components/num_components

def make_csim_figures(path_data, path_figures):
    df_c = pd.read_csv(f"{path_data}/alg_components.csv", index_col=0).T

    # filter
    unwanted_cols = ['Sampling', 'Ranking ', 'Step-size Adaptation']
    unwanted_rows = [] #algs
    df_c = df_c.drop(unwanted_cols, axis=0, inplace=False)
    df_c = df_c.drop(unwanted_rows, axis=1, inplace=False)

    df_cs = pd.DataFrame([], columns=df_c.columns, index=df_c.columns)
    for a, b in itertools.product(df_c.columns, df_c.columns):
        df_cs[b][a] = calc_component_similarity(df_c[a], df_c[b])
        
    df_cs = df_cs.applymap(lambda x: float("{:.02f}".format(x)))
    
    fig = make_heatmap(heat_data=df_cs, labels=df_cs.index)
    fig.write_image(f"{path_figures}/component_sim.png", scale=5)

    return df_cs

def make_heatmap(heat_data, labels):
    heatmap = go.Figure(data=go.Heatmap(z=heat_data, x=labels, y=labels,
                        colorscale = 'Blues',
                        text=pd.DataFrame(heat_data).applymap(lambda v: f'{v:.2f}'),
                        texttemplate="%{text}",
                        textfont={"size":16},
                        zmin=0, zmax=1, zauto=False
                       ))
    return heatmap

def calc_performance_similarity(data_array):
    # compute the performance similarity
    data_dist = pdist(data_array)
    heat_data = squareform(data_dist)
    heat_data = np.array(list(map(lambda v: 1/(1+v), heat_data)))  #scale 0 to 1
    return heat_data

def make_main_figures(data, myorder, path_fig):
    # we want 6 figures where figure 1 aggregates the performance similarity for all benchmark functions
    # and figures 2-6 are the performance similarity by function group

    # slices: how to separate the data function group
    slices = slice(0, 25), slice(0, 5), slice(5, 9), slice(9, 14), slice(14, 19), slice(19, 25)
    slices_name = "Functions 1-24", "Functions 1-5", "Functions 6-9", "Functions 10-14", "Functions 15-19", "Functions 20-24"

    # record performance similarity for each case
    d_psim = {}

    for (dim, metric), df in data.items():
        # rewording for figure title and file name
        if metric=="log10ert_area":
            metric="ERT Area"
        else:
            metric="Error"

        df = df[myorder] # desired algorithm order
        d_psim[(dim, metric)] = {}

        # for each figure (based on the above slices)
        for sname, sidx in zip(slices_name, slices):
            # define fig title and file name
            fig_title = f"{sname} in {dim}D using the {metric} metric"
            fig_name  = f"{dim}D_{metric}_{sname}.png"
            
            # compute performance similarity
            dftemp = df[sidx].T
            data_array = dftemp.values*dim # rescale based off on problems dimension

            heat_data = calc_performance_similarity(data_array=data_array)
            
            # create heatmap
            fig = make_heatmap(heat_data, labels=dftemp.index)

            # data structure for pearson correlation
            d_psim[(dim, metric)][sname] = pd.DataFrame(heat_data, columns=dftemp.index, index=dftemp.index)

            # adjust figure settings
            fig.update_layout(title={'text': f"{fig_title}",
                                     'y':0.9,
                                     'x':0.5,
                                     'xanchor': 'center',
                                     'yanchor': 'top'})
            fig.update_layout(font=dict(family="Arial")) #true type font

            fig.write_image(f"{path_fig}/{fig_name}", scale=5)

    return d_psim

def get_data_metric(df, metric):
    # create a df with only the metrics of interest 
    newdf = pd.DataFrame([], columns=df.algorithm.unique(), index=df.function.unique())

    for alg in df.algorithm.unique():
        mask = (df.algorithm==alg)
        newdf[alg] = df[mask][metric].values
        
    if metric == "error_avg": # if the metric is error_avg, apply log10
        newdf = newdf.applymap(lambda v: -8 if v<=1e-8 else np.log10(v))
        
    return newdf

def load_data(dimensions, metrics, path):
    # create the performance profile table for the chosen algorithm divided by dimension / metric
    data = {}
    for dim in dimensions:
        df_tmp = pd.read_csv(f"{path}/bbob_fall_{dim}d.csv")
        
        for m in metrics:
            df_tmp2 = get_data_metric(df_tmp, m)
            alg_list = df_tmp.algorithm.unique()
            data[(dim, m)] = df_tmp2.filter(regex="|".join(alg_list),axis=1)

    #data[(dimensions[0], metrics[0])].head()
    return data

def calc_pearson(df_csim, df_psim, path):
    slices_name = "Functions 1-24", "Functions 1-5", "Functions 6-9", "Functions 10-14", "Functions 15-19", "Functions 20-24"

    # component similarity
    y = df_csim.values.flatten()

    # pearson similarity per group of performance similarity
    d_corr = {}
    for (dim, metric), d in df_psim.items(): #todo: test if it works for more than one dim, and more than one metric
        d_corr[(dim, metric)] = pd.DataFrame([], columns=slices_name, index=["corr", "pvalue"])    
        for name, v in d.items():
            x = v.drop("RANDOMSEARCH", axis=0).drop("RANDOMSEARCH", axis=1).values.flatten()

            c, p = stats.pearsonr(x, y)
            d_corr[(dim, metric)][name]["corr"]   = c
            d_corr[(dim, metric)][name]["pvalue"] = p
    
        d_corr[(dim, metric)].T.to_csv(f"{path}/pearson_{metric}_{dim}D.csv", index=False)

def main(args):
    # component similarity
    df_csim = make_csim_figures(path_data=args["PATH_COMP"], path_figures=args["PATH_FIG"])

    # performance similarity
    data = load_data(dimensions=args["DIMENSIONS"], metrics=args["METRICS"], path=args["PATH_CSV"])
    df_psim = make_main_figures(data=data, myorder=args["MYORDER"], path_fig=args["PATH_FIG"])

    # pearson correlation
    calc_pearson(df_csim, df_psim, path=args["PATH_CORR"])
    
if __name__ == "__main__":
    if not os.path.exists(args["PATH_FIG"]):
        os.mkdir(args["PATH_FIG"])
    if not os.path.exists(args["PATH_CORR"]):
        os.mkdir(args["PATH_CORR"])

    main(args)



