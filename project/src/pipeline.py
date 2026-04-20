# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import yaml
from pathlib import Path

# --- Scikit-learn: Сборка пайплайна ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Локальные компоненты и трансформеры ---
from .config import TRAIN_PARAMS_DIR
from .transformers import (
    MissingValueImputer, 
    CombinedFeaturesAdded, 
    OrdinalEncoderTransformer
)

def get_pipeline(model=None):
    with open(os.path.join(TRAIN_PARAMS_DIR, 'data_params.yaml'), 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    ord_mapping = params['ordinal_mappings']
    ord_cols = list(ord_mapping.keys())
 
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoderTransformer(ord_mapping), ord_cols), 
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
             make_column_selector(dtype_include=['object', 'category', 'string']))
        ],
        remainder='passthrough' 
    )
    
    steps = [
        ('imputer', MissingValueImputer()),
        ('features', CombinedFeaturesAdded()),
        ('encoding', preprocessor),           
        ('scaler', StandardScaler())
    ]

    if model is not None:
        steps.append(('model', model))

    return Pipeline(steps=steps)