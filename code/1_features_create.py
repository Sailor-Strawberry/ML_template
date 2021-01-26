import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from base import Feature, get_arguments, generate_features
from sklearn.preprocessing import LabelEncoder
import warnings
import jpholiday
import optuna.integration.lightgbm as lgb
from datetime import datetime, date, timedelta
import calendar
from functools import partial

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE) as file:
    yml = yaml.load(stream=file, Loader=yaml.SafeLoader)

RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # 特徴量生成元のRAWデータ格納場所
Feature.dir = yml['SETTING']['FEATURE_PATH']  # 生成した特徴量の出力場所
feature_memo_path = Feature.dir + '_features_memo.csv'


# 循環特徴量生成用(2021_01_10)
def make_harmonic_features_cos(value, period):
    value *= 2 * np.pi / period 
    return np.cos(value)

def make_harmonic_features_sin(value, period):
    value *= 2 * np.pi / period 
    return np.sin(value)


# Target
class year(Feature):
    def create_features(self):
        self.train['year'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').dt.year.astype('int8')
        self.test['year'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').dt.year.astype('int8')
        create_memo('year','年')

class month(Feature):
    def create_features(self):
        self.train['month'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').dt.month.astype('int8')
        self.test['month'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').dt.month.astype('int8')
        create_memo('month','月')

class day(Feature):
    def create_features(self):
        self.train['day'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').dt.day.astype('int8')
        self.test['day'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').dt.day.astype('int8')
        create_memo('day','日')

class weekday(Feature):
    def create_features(self):
        self.train['weekday'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').dt.dayofweek.astype('int8')
        self.test['weekday'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').dt.dayofweek.astype('int8')
        create_memo('weekday','月曜日を０、日曜日を６')

class is_holiday(Feature):
    def create_features(self):
        self.train['is_holiday'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').map(jpholiday.is_holiday).astype('int8')
        self.test['is_holiday'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').map(jpholiday.is_holiday).astype('int8')
        create_memo('is_holiday','祝日なら「1」、そうでないなら「0」')

class before_is_holiday(Feature):
    def create_features(self):
        self.train['before_is_holiday'] = (((pd.to_datetime(train['datetime'],format='%Y-%m-%d')+timedelta(days=-1)).map(jpholiday.is_holiday)) | ((pd.to_datetime(train['datetime'],format='%Y-%m-%d')+timedelta(days=1)).dt.dayofweek>4)).astype(int)
        self.test['before_is_holiday'] = (((pd.to_datetime(test['datetime'],format='%Y-%m-%d')+timedelta(days=-1)).map(jpholiday.is_holiday)) | ((pd.to_datetime(test['datetime'],format='%Y-%m-%d')+timedelta(days=1)).dt.dayofweek>4)).astype(int)
        create_memo('before_is_holiday','前日が休日なら「1」、平日なら「0」')

class next_is_holiday(Feature):
    def create_features(self):
        self.train['next_is_holiday'] = (((pd.to_datetime(train['datetime'],format='%Y-%m-%d')+timedelta(days=1)).map(jpholiday.is_holiday)) | ((pd.to_datetime(train['datetime'],format='%Y-%m-%d')+timedelta(days=1)).dt.dayofweek>4)).astype(int)
        self.test['next_is_holiday'] = (((pd.to_datetime(test['datetime'],format='%Y-%m-%d')+timedelta(days=1)).map(jpholiday.is_holiday)) | ((pd.to_datetime(test['datetime'],format='%Y-%m-%d')+timedelta(days=1)).dt.dayofweek>4)).astype(int)
        create_memo('next_is_holiday','翌日が休日なら「1」、平日なら「0」')

class dayofyear_sin(Feature):
    def create_features(self):
        dow_make_features_sin = partial(make_harmonic_features_sin, period=366)
        self.train['dayofyear_sin'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').dt.dayofyear.apply(dow_make_features_sin).astype('float32')
        self.test['dayofyear_sin'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').dt.dayofyear.apply(dow_make_features_sin).astype('float32')
        create_memo('dayofyear_sin','月の循環化')

class dayofyear_cos(Feature):
    def create_features(self):
        dow_make_features_cos = partial(make_harmonic_features_cos, period=366)
        self.train['dayofyear_cos'] = pd.to_datetime(train['datetime'],format='%Y-%m-%d').dt.dayofyear.apply(dow_make_features_cos).astype('float32')
        self.test['dayofyear_cos'] = pd.to_datetime(test['datetime'],format='%Y-%m-%d').dt.dayofyear.apply(dow_make_features_cos).astype('float32')
        create_memo('dayofyear_cos','月の循環化')

class client(Feature):
    def create_features(self):
        self.train['client'] = train['client']
        self.test['client'] = test['client']
        create_memo('client','法人が絡む特殊な引越し日フラグ')

class close(Feature):
    def create_features(self):
        self.train['close'] = train['close']
        self.test['close'] = test['close']
        create_memo('close','休業日')

class price_am(Feature):
    def create_features(self):
        self.train['price_am'] = train['price_am']
        self.test['price_am'] = test['price_am']
        create_memo('price_am','午前の料金区分（-1は欠損を表す。5が最も料金が高い）')

class price_pm(Feature):
    def create_features(self):
        self.train['price_pm'] = train['price_pm']
        self.test['price_pm'] = test['price_pm']
        create_memo('price_pm','午後の料金区分（-1は欠損を表す。5が最も料金が高い）')

class y(Feature):
    def create_features(self):
        self.train['y'] = train['y']
        create_memo('y','引っ越し数、目的変数')



# 特徴量メモcsvファイル作成
def create_memo(col_name, desc):

    file_path = Feature.dir + '/_features_memo.csv'
    if not os.path.isfile(file_path):
        with open(file_path,"w"):pass

    with open(file_path, 'r+') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(',')[0] == col_name]
        if len(col) != 0:return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])

if __name__ == '__main__':

    # CSVのヘッダーを書き込み
    create_memo('特徴量', 'メモ')

    args = get_arguments()
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')

    # globals()でtrain,testのdictionaryを渡す
    generate_features(globals(), args.force)

    # 特徴量メモをソートする
    feature_df = pd.read_csv(feature_memo_path)
    feature_df = feature_df.sort_values('特徴量')
    feature_df.to_csv(feature_memo_path, index=False)
