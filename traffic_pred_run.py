#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv('dataset/train.csv')
test=pd.read_csv('dataset/test.csv')


# In[3]:


train.sample(5)


# In[4]:


test.sample(5)


# In[5]:


train.dtypes


# In[6]:


test.dtypes


# In[7]:


train.isnull().sum()
test.isnull().sum() ## hence some null values


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


train.describe()


# In[11]:


test.describe()


# In[12]:


## decoding geohash into readble number


# In[13]:


base32 = '0123456789bcdefghjkmnpqrstuvwxyz'


# In[14]:


all_geohashes = list(set(list(train['geohash']) + list(test['geohash'])))
all_geohashes


# In[15]:


gh_lat = {}
gh_lon = {}


# In[16]:


for gh in all_geohashes:
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        cd = base32.index(ch)
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    gh_lat[gh] = (lat_lo + lat_hi) / 2
    gh_lon[gh] = (lon_lo + lon_hi) / 2


# In[17]:


len(gh_lat)


# In[18]:


gh_lon


# In[19]:


for df in [train, test]:
    # timestamp like "2:15" -> minutes since midnight
    df['tmin'] = df['timestamp'].str.split(':').str[0].astype(int) * 60 \
               + df['timestamp'].str.split(':').str[1].astype(int)
    df['hour'] = df['tmin'] // 60
    df['tod_bucket'] = df['tmin'] // 30          # 30-minute slot of the day
    df['quarter'] = df['tmin'] // 15              # finer granularity

    # cyclical time: tells the model that 23:45 and 00:00 are close together
    df['min_sin']  = np.sin(2 * np.pi * df['tmin'] / 1440)
    df['min_cos']  = np.cos(2 * np.pi * df['tmin'] / 1440)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # location from the decoded geohash
    df['lat'] = df['geohash'].map(gh_lat)
    df['lon'] = df['geohash'].map(gh_lon)

    # coarser region codes + location-time combos
    df['gh_prefix5'] = df['geohash'].str[:5]
    df['gh_prefix4'] = df['geohash'].str[:4]
    df['gh_prefix3'] = df['geohash'].str[:3]
    df['gh_hour']    = df['geohash'] + '_' + df['hour'].astype(str)
    df['gh_bucket']  = df['geohash'] + '_' + df['tod_bucket'].astype(str)
    df['gh_quarter'] = df['geohash'] + '_' + df['quarter'].astype(str)
    df['prefix5_hour'] = df['gh_prefix5'] + '_' + df['hour'].astype(str)

    # Peak hour flags
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)


# In[20]:


train['temp_missing'] = train['Temperature'].isna().astype(int)
test['temp_missing']  = test['Temperature'].isna().astype(int) ## figuring out null values in temp in a column


# In[21]:


train.sample(5)


# In[22]:


temp_by_gh = train.groupby('geohash')['Temperature'].median()
glob_temp  = train['Temperature'].median()
train['Temperature'] = train['Temperature'].fillna(train['geohash'].map(temp_by_gh)).fillna(glob_temp)
test['Temperature']  = test['Temperature'].fillna(test['geohash'].map(temp_by_gh)).fillna(glob_temp) ## filling the missing temp values with median


# In[23]:


train.sample(5)


# In[24]:


for df in [train, test]:
    df['LargeVehicles_e'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_e']     = (df['Landmarks'] == 'Yes').astype(int)

road_dummies = pd.get_dummies(pd.concat([train['RoadType'], test['RoadType']]), prefix='road', dummy_na=True)
train_road = road_dummies.iloc[:len(train)].reset_index(drop=True)
test_road  = road_dummies.iloc[len(train):].reset_index(drop=True)
for c in train_road.columns:
    train[c] = train_road[c].astype(int)
    test[c]  = test_road[c].astype(int)

weather_dummies = pd.get_dummies(pd.concat([train['Weather'], test['Weather']]), prefix='weather', dummy_na=True)
train_weather = weather_dummies.iloc[:len(train)].reset_index(drop=True)
test_weather  = weather_dummies.iloc[len(train):].reset_index(drop=True)
for c in train_weather.columns:
    train[c] = train_weather[c].astype(int)
    test[c]  = test_weather[c].astype(int)

road_cols = [c for c in train.columns if c.startswith('road_')]
weather_cols = [c for c in train.columns if c.startswith('weather_')]


# In[25]:


for df in [train, test]:
    df['temp_x_hour']    = df['Temperature'] * df['hour']
    df['lanes_x_road']   = df['NumberofLanes'] * df['LargeVehicles_e']
    df['lanes_x_hour']   = df['NumberofLanes'] * df['hour']
    df['lat_x_lon']      = df['lat'] * df['lon']
    df['temp_x_lanes']   = df['Temperature'] * df['NumberofLanes']
    df['rush_x_lanes']   = (df['is_morning_rush'] + df['is_evening_rush']) * df['NumberofLanes']
    df['temp_sq']        = df['Temperature'] ** 2
    df['lat_x_hour']     = df['lat'] * df['hour']
    df['lon_x_hour']     = df['lon'] * df['hour']

    # NEW interaction features
    df['temp_x_landmark'] = df['Temperature'] * df['Landmarks_e']
    df['temp_x_large']    = df['Temperature'] * df['LargeVehicles_e']
    df['hour_sq']         = df['hour'] ** 2
    df['tmin_sq']         = df['tmin'] ** 2
    df['lat_sq']          = df['lat'] ** 2
    df['lon_sq']          = df['lon'] ** 2
    df['lanes_sq']        = df['NumberofLanes'] ** 2
    df['rush_any']        = ((df['is_morning_rush'] + df['is_evening_rush']) > 0).astype(int)
    df['rush_x_large']    = df['rush_any'] * df['LargeVehicles_e']
    df['rush_x_landmark'] = df['rush_any'] * df['Landmarks_e']
    df['night_x_lanes']   = df['is_night'] * df['NumberofLanes']
    df['night_x_temp']    = df['is_night'] * df['Temperature']
    df['lat_x_tmin']      = df['lat'] * df['tmin']
    df['lon_x_tmin']      = df['lon'] * df['tmin']

    # Geohash string features as numeric
    df['gh_char0'] = df['geohash'].str[0].map({c: i for i, c in enumerate(base32)})
    df['gh_char5'] = df['geohash'].str[5].map({c: i for i, c in enumerate(base32)})

# Frequency encoding (no target leakage)
gh_freq = train['geohash'].value_counts().to_dict()
p5_freq = train['gh_prefix5'].value_counts().to_dict()
p4_freq = train['gh_prefix4'].value_counts().to_dict()
for df in [train, test]:
    df['gh_freq']  = df['geohash'].map(gh_freq).fillna(0)
    df['p5_freq']  = df['gh_prefix5'].map(p5_freq).fillna(0)
    df['p4_freq']  = df['gh_prefix4'].map(p4_freq).fillna(0)


# In[26]:


train.sample(5)


# In[27]:


# pip install lightgbm catboost


# In[28]:


from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor


# In[29]:


train['y'] = np.log1p(train['demand'])
GM = train['demand'].mean()

te_col_defs = {
    'geohash': 10, 'gh_prefix5': 15, 'gh_prefix4': 20, 'gh_prefix3': 25,
    'hour': 30, 'tmin': 30, 'gh_hour': 8, 'gh_bucket': 6,
    'tod_bucket': 30, 'gh_quarter': 5, 'prefix5_hour': 12, 'quarter': 25
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# OOF target encoding for train
for col, m in te_col_defs.items():
    oof = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(col)['demand'].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + m * GM) / (agg['count'] + m)
        oof[val_idx] = train.iloc[val_idx][col].map(smooth).fillna(GM).values
    train['te_' + col] = oof
    agg = train.groupby(col)['demand'].agg(['count', 'mean'])
    smooth = (agg['count'] * agg['mean'] + m * GM) / (agg['count'] + m)
    test['te_' + col] = test[col].map(smooth).fillna(GM).values

# NEW: Compound target encodings (geohash × categorical)
for df in [train, test]:
    df['gh_weather']  = df['geohash'] + '_' + df['Weather'].fillna('NA')
    df['gh_road']     = df['geohash'] + '_' + df['RoadType'].fillna('NA')
    df['gh_large']    = df['geohash'] + '_' + df['LargeVehicles']
    df['p5_bucket']   = df['gh_prefix5'] + '_' + df['tod_bucket'].astype(str)
    df['gh_rush']     = df['geohash'] + '_' + df['rush_any'].astype(str)

extra_te = {
    'gh_weather': 5, 'gh_road': 5, 'gh_large': 5,
    'p5_bucket': 8, 'gh_rush': 5
}
for col, m in extra_te.items():
    oof = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(col)['demand'].agg(['count', 'mean'])
        smooth = (agg['count'] * agg['mean'] + m * GM) / (agg['count'] + m)
        oof[val_idx] = train.iloc[val_idx][col].map(smooth).fillna(GM).values
    train['te_' + col] = oof
    agg = train.groupby(col)['demand'].agg(['count', 'mean'])
    smooth = (agg['count'] * agg['mean'] + m * GM) / (agg['count'] + m)
    test['te_' + col] = test[col].map(smooth).fillna(GM).values

# OOF aggregation features (leak-free)
for col_name, agg_func, fill_val, feat_name in [
    ('geohash', 'std', 0, 'te_geohash_std'),
    ('geohash', lambda x: x.quantile(0.5), GM, 'gh_med'),
    ('geohash', lambda x: x.quantile(0.25), GM, 'gh_q25'),
    ('geohash', lambda x: x.quantile(0.75), GM, 'gh_q75'),
    ('geohash', lambda x: x.quantile(0.1), GM, 'gh_q10'),
    ('geohash', lambda x: x.quantile(0.9), GM, 'gh_q90'),
    ('geohash', 'max', GM, 'gh_max'),
    ('geohash', 'min', 0, 'gh_min'),
    ('gh_prefix5', 'std', 0, 'p5_std'),
    ('gh_prefix5', lambda x: x.quantile(0.5), GM, 'p5_med'),
]:
    oof = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(col_name)['demand'].agg(agg_func)
        if isinstance(agg, pd.Series):
            agg = agg.fillna(fill_val)
        oof[val_idx] = train.iloc[val_idx][col_name].map(agg).fillna(fill_val).values
    train[feat_name] = oof
    agg_full = train.groupby(col_name)['demand'].agg(agg_func)
    if isinstance(agg_full, pd.Series):
        agg_full = agg_full.fillna(fill_val)
    test[feat_name] = test[col_name].map(agg_full).fillna(fill_val)

# OOF gh_hour_med, gh_hour_std, and gh_bucket_med
for group_cols, agg_name, fill_val, feat in [
    (['geohash', 'hour'], 'median', GM, 'gh_hour_med'),
    (['geohash', 'hour'], 'std', 0, 'gh_hour_std'),
    (['geohash', 'tod_bucket'], 'median', GM, 'gh_bucket_med'),
    (['geohash', 'tod_bucket'], 'std', 0, 'gh_bucket_std'),
]:
    oof = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(group_cols)['demand'].agg(agg_name)
        lookup = train.iloc[val_idx].set_index(group_cols).index
        oof[val_idx] = [agg.get(k, fill_val) for k in lookup]
    train[feat] = oof
    agg_full = train.groupby(group_cols)['demand'].agg(agg_name)
    lookup_test = test.set_index(group_cols).index
    test[feat] = [agg_full.get(k, fill_val) for k in lookup_test]

# Demand range feature per geohash
oof_range = np.zeros(len(train))
for tr_idx, val_idx in kf.split(train):
    rng = train.iloc[tr_idx].groupby('geohash')['demand'].agg(lambda x: x.max() - x.min())
    oof_range[val_idx] = train.iloc[val_idx]['geohash'].map(rng).fillna(0).values
train['gh_range'] = oof_range
rng_full = train.groupby('geohash')['demand'].agg(lambda x: x.max() - x.min())
test['gh_range'] = test['geohash'].map(rng_full).fillna(0)

# Feature list
te_feature_names = ['te_' + col for col in te_col_defs.keys()]
extra_te_names = ['te_' + col for col in extra_te.keys()]
agg_feat_names = [
    'te_geohash_std', 'gh_med', 'gh_q25', 'gh_q75', 'gh_q10', 'gh_q90',
    'gh_max', 'gh_min', 'p5_std', 'p5_med',
    'gh_hour_med', 'gh_hour_std', 'gh_bucket_med', 'gh_bucket_std', 'gh_range'
]

new_interaction_feats = [
    'temp_x_landmark', 'temp_x_large', 'hour_sq', 'tmin_sq',
    'lat_sq', 'lon_sq', 'lanes_sq', 'rush_any', 'rush_x_large',
    'rush_x_landmark', 'night_x_lanes', 'night_x_temp',
    'lat_x_tmin', 'lon_x_tmin', 'gh_char0', 'gh_char5',
    'p5_freq', 'p4_freq'
]

features = [
    'day', 'tmin', 'hour', 'tod_bucket', 'quarter',
    'min_sin', 'min_cos', 'hour_sin', 'hour_cos',
    'lat', 'lon', 'NumberofLanes', 'Temperature', 'temp_missing',
    'LargeVehicles_e', 'Landmarks_e',
    'is_morning_rush', 'is_evening_rush', 'is_night',
    'gh_freq',
    'temp_x_hour', 'lanes_x_road', 'lanes_x_hour', 'lat_x_lon',
    'temp_x_lanes', 'rush_x_lanes', 'temp_sq', 'lat_x_hour', 'lon_x_hour',
] + new_interaction_feats + road_cols + weather_cols + te_feature_names + extra_te_names + agg_feat_names

print(f"Total features: {len(features)}")


# In[30]:


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

X_all = train[features]
y_all = train['y']

oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_cb  = np.zeros(len(train))
oof_et  = np.zeros(len(train))
test_xgb = np.zeros(len(test))
test_lgb = np.zeros(len(test))
test_cb  = np.zeros(len(test))
test_et  = np.zeros(len(test))


# In[ ]:


for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n=== Fold {fold} ===")
    X_f_tr, X_f_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_f_tr, y_f_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

    # XGBoost - deeper, more trees, lower LR
    m_xgb = xgb.XGBRegressor(
        n_estimators=8000, learning_rate=0.008,
        max_depth=10, min_child_weight=3,
        subsample=0.75, colsample_bytree=0.55, colsample_bylevel=0.7,
        reg_alpha=0.3, reg_lambda=1.5,
        gamma=0.05, random_state=42, n_jobs=-1,
        early_stopping_rounds=200, eval_metric='rmse'
    )
    m_xgb.fit(X_f_tr, y_f_tr, eval_set=[(X_f_val, y_f_val)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_f_val)
    test_xgb += m_xgb.predict(test[features]) / 5

    # LightGBM - more leaves, lower LR
    m_lgb = lgb.LGBMRegressor(
        n_estimators=8000, learning_rate=0.008,
        max_depth=10, num_leaves=300,
        min_child_samples=15,
        subsample=0.75, colsample_bytree=0.55,
        reg_alpha=0.3, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    m_lgb.fit(X_f_tr, y_f_tr, eval_set=[(X_f_val, y_f_val)],
              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = m_lgb.predict(X_f_val)
    test_lgb += m_lgb.predict(test[features]) / 5

    # CatBoost - deeper
    m_cb = CatBoostRegressor(
        iterations=8000, learning_rate=0.008,
        depth=10, l2_leaf_reg=1.5,
        subsample=0.75, random_seed=42, verbose=0,
        early_stopping_rounds=200
    )
    m_cb.fit(X_f_tr, y_f_tr, eval_set=(X_f_val, y_f_val))
    oof_cb[val_idx] = m_cb.predict(X_f_val)
    test_cb += m_cb.predict(test[features]) / 5

    # ExtraTrees for diversity
    m_et = ExtraTreesRegressor(
        n_estimators=500, max_depth=25, min_samples_leaf=3,
        max_features=0.6, random_state=42, n_jobs=-1
    )
    m_et.fit(X_f_tr, y_f_tr)
    oof_et[val_idx] = m_et.predict(X_f_val)
    test_et += m_et.predict(test[features]) / 5

    v_xgb = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_xgb[val_idx]), 1e-7, 1))
    v_lgb = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_lgb[val_idx]), 1e-7, 1))
    v_cb  = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_cb[val_idx]), 1e-7, 1))
    v_et  = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_et[val_idx]), 1e-7, 1))
    print(f"  XGB={v_xgb:.4f}  LGB={v_lgb:.4f}  CB={v_cb:.4f}  ET={v_et:.4f}")


# In[ ]:


p_oof_xgb = np.clip(np.expm1(oof_xgb), 1e-7, 1.0)
p_oof_lgb = np.clip(np.expm1(oof_lgb), 1e-7, 1.0)
p_oof_cb  = np.clip(np.expm1(oof_cb), 1e-7, 1.0)
p_oof_et  = np.clip(np.expm1(oof_et), 1e-7, 1.0)
y_true = train['demand'].values

print("OOF XGB R2:", round(r2_score(y_true, p_oof_xgb), 4))
print("OOF LGB R2:", round(r2_score(y_true, p_oof_lgb), 4))
print("OOF CB  R2:", round(r2_score(y_true, p_oof_cb), 4))
print("OOF ET  R2:", round(r2_score(y_true, p_oof_et), 4))


# In[ ]:


# ---- Stacking: Ridge meta-learner on OOF predictions ----
oof_stack = np.column_stack([p_oof_xgb, p_oof_lgb, p_oof_cb, p_oof_et])

# Cross-validated stacking to avoid overfitting the meta-learner
meta_oof = np.zeros(len(train))
meta_test_preds = np.zeros(len(test))
test_stack = np.column_stack([
    np.clip(np.expm1(test_xgb), 1e-7, 1.0),
    np.clip(np.expm1(test_lgb), 1e-7, 1.0),
    np.clip(np.expm1(test_cb), 1e-7, 1.0),
    np.clip(np.expm1(test_et), 1e-7, 1.0),
])

for tr_idx, val_idx in kf.split(oof_stack):
    meta = Ridge(alpha=1.0)
    meta.fit(oof_stack[tr_idx], y_true[tr_idx])
    meta_oof[val_idx] = meta.predict(oof_stack[val_idx])
    meta_test_preds += meta.predict(test_stack) / 5

stacked_r2 = r2_score(y_true, np.clip(meta_oof, 1e-7, 1.0))
print(f"Stacked OOF R2: {stacked_r2:.4f}")

# Also try simple grid search for comparison
best_r2 = -1
best_w = None
for w1 in np.arange(0.0, 0.5, 0.05):
    for w2 in np.arange(0.0, 0.6, 0.05):
        for w3 in np.arange(0.0, 0.8, 0.05):
            w4 = 1 - w1 - w2 - w3
            if w4 < 0: continue
            p_ens = w1 * p_oof_xgb + w2 * p_oof_lgb + w3 * p_oof_cb + w4 * p_oof_et
            r2 = r2_score(y_true, p_ens)
            if r2 > best_r2:
                best_r2 = r2
                best_w = (round(w1, 2), round(w2, 2), round(w3, 2), round(w4, 2))

print(f"\nBest grid weights: XGB={best_w[0]}, LGB={best_w[1]}, CB={best_w[2]}, ET={best_w[3]}")
print(f"Grid Ensemble R2: {best_r2:.4f}")

# Use whichever is better
use_stacking = stacked_r2 > best_r2
final_r2 = max(stacked_r2, best_r2)
score = max(0, 100 * final_r2)
print(f"\n>>> Using {'stacking' if use_stacking else 'grid weights'}")
print(f">>> Competition Score = max(0, 100 * R2) = {score:.2f}")


# In[ ]:


t_xgb = np.clip(np.expm1(test_xgb), 1e-7, 1.0)
t_lgb = np.clip(np.expm1(test_lgb), 1e-7, 1.0)
t_cb  = np.clip(np.expm1(test_cb), 1e-7, 1.0)
t_et  = np.clip(np.expm1(test_et), 1e-7, 1.0)

if use_stacking:
    test['demand'] = np.clip(meta_test_preds, 1e-7, 1.0)
else:
    test['demand'] = best_w[0] * t_xgb + best_w[1] * t_lgb + best_w[2] * t_cb + best_w[3] * t_et

submission = test[['Index', 'demand']]
submission.to_csv('submission.csv', index=False)

print(f"Submission shape: {submission.shape}")
print(f"Columns: {list(submission.columns)}")
print(f"Demand range: [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
print(f"\nSubmission saved to submission.csv")


# In[ ]:




