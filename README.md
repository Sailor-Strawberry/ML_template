## 動作検証済み環境
OS: MacOS BigSur  
python: 3.8.3

# 手順

## クローン
```sh
git clone https://github.com/Sailor-Strawberry/ml_pipeline
```

## フォルダ移動
```sh
cd ml_pipeline/code
```

## 特徴量生成
```sh
python 1_features_create.py
```

## 生成された特徴量の確認（確認したい場合）
```sh
python 2_features_verification.py
```

## 学習
```sh
python 3_run.py
```

## 特徴量生成テンプレート(通常)
```sh
class hoge(Feature):
    def create_features(self):
        self.train['hoge'] = train['hoge']
        self.test['hoge'] = test['hoge']
        create_memo('hoge','ここに変数の説明を入力')
```

## 特徴量生成テンプレート(LabelEncoding)
```sh
class hoge(Feature):
    def create_features(self):
        cols = 'hoge'
        tmp_df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
        le = LabelEncoder().fit(tmp_df[cols])
        self.train['hoge'] = le.transform(train[cols])
        self.test['hoge'] = le.transform(test[cols])
        create_memo('hoge','ここに変数の説明を入力')
```

## 特徴量生成テンプレート(循環特徴量)
```sh
class hoge(Feature):
    def create_features(self):
        dow_make_features_cos = partial(make_harmonic_features_cos, period=12)
        self.train['hoge'] = pd.to_datetime(train['date'],format='%Y-%m-%d').dt.month.apply(dow_make_features_cos).astype('float32')
        self.test['hoge'] = pd.to_datetime(test['date'],format='%Y-%m-%d').dt.month.apply(dow_make_features_cos).astype('float32')
        create_memo('hoge','月の循環化')
```

## 特徴量生成テンプレート(前日フラグ)
```sh
class before_is_holiday(Feature):
    def create_features(self):
        self.train['before_is_holiday'] = (((pd.to_datetime(train['datetime'],format='%Y-%m-%d')+timedelta(days=-1)).map(jpholiday.is_holiday))| ((pd.to_datetime(train['datetime'],format='%Y-%m-%d')+timedelta(days=1)).dt.dayofweek>4)).astype(int)
        self.test['before_is_holiday'] = (((pd.to_datetime(test['datetime'],format='%Y-%m-%d')+timedelta(days=-1)).map(jpholiday.is_holiday)) | ((pd.to_datetime(test['datetime'],format='%Y-%m-%d')+timedelta(days=1)).dt.dayofweek>4)).astype(int)
        create_memo('before_is_holiday','前日が休日なら「1」、平日なら「0」')
```