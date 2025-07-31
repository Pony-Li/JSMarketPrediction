import numpy as np
import polars as pl

def calculate_r2(y_true, y_pred, sample_weight):
    """
    计算加权 R² 分数
    """
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / \
              (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2


def encode_column(df, column, mapping):
    """
    对某一列按给定 mapping 进行编码
    """
    max_value = max(mapping.values())  

    def encode_category(category):
        return mapping.get(category, max_value + 1)
    
    return df.with_columns(
        pl.col(column).map_elements(encode_category).alias(column)
    )
