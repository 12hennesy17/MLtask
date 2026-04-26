# -*- coding: utf-8 -*- #говорим явно читать как UTF-8
from __future__ import annotations

import os
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- Локальные импорты ---
from . import config

class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Сохраняем моды и медианы, чтобы использовать их на тесте 
        with open(os.path.join(config.TRAIN_PARAMS_DIR, 'data_params.yaml') , 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        self.impute_config = params['imputation']
        self.lot_frontage_medians = None
        self.global_median_frontage = None
        self.modes = {}

    def fit(self, X, y=None):
        # Считаем медиану LotFrontage по районам
        self.lot_frontage_medians = X.groupby('Neighborhood')['LotFrontage'].median()
        self.global_median_frontage = X['LotFrontage'].median()
        self.feature_names_in_ = X.columns.tolist()

        # Считаем моды для колонок из конфига
        for col in self.impute_config['mode_cols']:
            if col in X.columns:
                self.modes[col] = X[col].mode()[0]

        return self

    def transform(self, X):
        X_copy = X.copy()
        cfg = self.impute_config

        # 1. Категориальные -> 'None' 
        for col in cfg['none_cols']:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna('None')

        # 2. Числовые -> 0
        for col in cfg['zero_cols']:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(0)

        # 3. LotFrontage (используем медианы из fit)
        X_copy['LotFrontage'] = X_copy.apply(
            lambda row: self.lot_frontage_medians.get(row['Neighborhood'], self.global_median_frontage)
            if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1
        )
        #заполняем самым частным значением в случае Nan
        for col, mode_val in self.modes.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].fillna(mode_val)

        # Если после всех правил остались NaN 
        remaining_na = X_copy.columns[X_copy.isnull().any()]
        for col in remaining_na:
            if X_copy[col].dtype in ['float64', 'int64']:
                X_copy[col] = X_copy[col].fillna(0) # Или глобальной медианой из fit
            else:
                # Для категорий берем моду из fit, если она есть, иначе 'None'
                fill_val = self.modes.get(col, 'None')
                X_copy[col] = X_copy[col].fillna(fill_val)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        # Если input_features нет, берем те, что запомнили в fit
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        # Если и там нет - это критическая ошибка
        if input_features is None:
            raise ValueError("Трансформер не обучен.")
        return np.array(input_features, dtype=object)
    
class CombinedFeaturesAdded(BaseEstimator, TransformerMixin):
  
    def __init__(self, drop_originals=True):
        self.drop_originals = drop_originals
        # Сразу определяем список новых фич, чтобы он был доступен везде
        self.new_features_ = [
            'IsNearRoad', 'IsNearRail', 'IsNearPosFeature', 'HasAlley',
            'TotalSF', 'TotalBath', 'LivingArea_per_Bedroom', 'IsNonStandardLayout',
            'HasPool', 'HasMiscFeature', 'HasFence', 
            'Has_LowQual_Area', 'IsNonStandardExterior',
            'HouseAge', 'YearsSinceRemod', 'MoSold_sin', 'MoSold_cos',
            'Is_New_Construction', 'Is_Distressed_Sale', 'IsBadCond', 'Has_Standard_Heating', 'IsNonStandardRoof'
        ]
        with open(os.path.join(config.TRAIN_PARAMS_DIR, 'data_params.yaml'), 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        
        # Раскладываем параметры по атрибутам класса
        self.fe_params = params['feature_engineering']
        self.cols_to_drop = params['cols_to_drop']
        
    def fit(self, X, y=None):        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        fe = self.fe_params
        # Локация и инфраструктура
        X_copy['IsNearRoad'] = X_copy['Condition1'].isin(fe['road_categories']).astype(int)
        X_copy['IsNearRail'] = X_copy['Condition1'].isin(fe['rail_categories']).astype(int)
        X_copy['IsNearPosFeature'] = X_copy['Condition1'].isin(fe['pos_features']).astype(int)
        X_copy['HasAlley'] = np.where(X_copy['Alley'] == 'None', 0, 1)
        X_copy['IsNonStandardRoof'] = (X_copy['RoofMatl'] != fe['standard_roof']).astype(int)
        # 2. Площади и комнаты
        X_copy['TotalSF'] = X_copy['GrLivArea'] + X_copy['TotalBsmtSF']
        X_copy['TotalBath'] = (X_copy['FullBath'] + (0.5 * X_copy['HalfBath']) +
                               X_copy['BsmtFullBath'] + (0.5 * X_copy['BsmtHalfBath']))
        X_copy['LivingArea_per_Bedroom'] = np.where(
            X_copy['BedroomAbvGr'] == 0, 
            X_copy['GrLivArea'], 
            X_copy['GrLivArea'] / X_copy['BedroomAbvGr']
        )
        X_copy['UnfBsmt_Ratio'] = np.where(
            X_copy['TotalBsmtSF'] > 0, 
            X_copy['BsmtUnfSF'] / X_copy['TotalBsmtSF'], 
            0
        )
        X_copy['IsNonStandardLayout'] = (X_copy['KitchenAbvGr'] != 1).astype(int)
      
        X_copy['Has3SsnPorch'] = (X_copy['3SsnPorch'] != 0).astype(int)

        # Качество и состояние
        X_copy['HasPool'] = np.where(X_copy['PoolQC'] == 'None', 0, 1)
        X_copy['HasMiscFeature'] = np.where(X_copy['MiscFeature'] == 'None', 0, 1)
        X_copy['HasFence'] = np.where(X_copy['Fence'] == 'None', 0, 1)

        X_copy['Has_LowQual_Area'] = (X_copy['LowQualFinSF'] > 0).astype(int)
      
        X_copy['IsBadCond'] = (X_copy['OverallCond'] < 5).astype(int)
        X_copy['IsNonStandardExterior'] = (X_copy['Exterior1st'] != X_copy['Exterior2nd']).astype(int)
        X_copy['Has_Standard_Heating'] = (X_copy['Heating'] != fe['standard_heating']).astype(int)
    
        # Временные признаки
        X_copy['HouseAge'] = X_copy['YrSold'] - X_copy['YearBuilt']
        X_copy['YearsSinceRemod'] = X_copy['YrSold'] - X_copy['YearRemodAdd']
        X_copy['MSSubClass'] = X_copy['MSSubClass'].astype(str)
        X_copy['MoSold_sin'] = np.sin(2 * np.pi * X_copy['MoSold'] / 12)
        X_copy['MoSold_cos'] = np.cos(2 * np.pi * X_copy['MoSold'] / 12)

        # Бизнес-фичи
        X_copy['Is_New_Construction'] = ((X_copy['SaleType'] == 'New') | 
                                        (X_copy['SaleCondition'] == 'Partial')).astype(int)
        
        X_copy['Is_Distressed_Sale'] = ((X_copy['SaleCondition'].isin(fe['distressed_conditions'])) | 
                                        (X_copy['SaleType'].isin(fe['distressed_types']))).astype(int)
        
        if self.drop_originals:
            existing_cols = [c for c in self.cols_to_drop if c in X_copy.columns]
            X_copy.drop(columns=existing_cols, inplace=True)
        self.feature_names_out_ = X_copy.columns.tolist()
        return X_copy  

    def get_feature_names_out(self, input_features=None):        
        if hasattr(self, 'feature_names_out_'):
            return np.array(self.feature_names_out_, dtype=object)
        
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
    
        if input_features is None:
            raise ValueError("Трансформер не обучен.")

       
        remaining = [f for f in input_features if f not in self.cols_to_drop]
        return np.array(remaining + self.new_features_, dtype=object)

class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_mappings):
        self.column_mappings = column_mappings

    def fit(self, X, y = None):
        self.feature_names_in_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.column_mappings.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping).fillna(0).astype(int)
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None


