import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro

def main():
    ref_mat = np.random.rand(20, 3) * 10
    props = np.array([
        [0.5, 0.2, 0.3],
        [0.1, 0.8, 0.1]
    ])
    query_mat = np.dot(ref_mat, props.T)
    
    ref_df = pd.DataFrame(
        data=ref_mat,
        index=[
            'gene_{}'.format(i) 
            for i in range(ref_mat.shape[0])
        ],
        columns=[
            'cell_type_{}'.format(i)
            for i in range(ref_mat.shape[1]) 
        ]
    )
    print(ref_df)

    query_df = pd.DataFrame(
        data=query_mat,
        index=[
            'gene_{}'.format(i)
            for i in range(query_mat.shape[0])
        ],
        columns=[
            'mixture_{}'.format(i)
            for i in range(query_mat.shape[1])
        ]
    )
    print(query_df)
    res = run_DeconRNASeq(ref_df, query_df)
    print(res)

def run_DeconRNASeq(ref_df, query_df):
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_ref_df = ro.conversion.py2rpy(ref_df)
        r_query_df = ro.conversion.py2rpy(query_df)

    rstring="""
    function(ref_df, query_df){
        library(DeconRNASeq)
        decon_r <- DeconRNASeq(
            query_df,
            ref_df,
            use.scale = FALSE
        )
        decon_r$out.all
    }
    """
    r_decon_func=robjects.r(rstring)
    r_decon_result = r_decon_func(r_ref_df, r_query_df)
    with localconverter(ro.default_converter + pandas2ri.converter):
        decon_result = ro.conversion.rpy2py(r_decon_result)

    #print(decon_result)
    result_df = pd.DataFrame(
        data=decon_result,
        index=query_df.columns,
        columns=ref_df.columns
    )
    return result_df

if __name__ == "__main__":
    main()
