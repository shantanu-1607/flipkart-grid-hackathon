import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ── Geohash decoding ──
base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
all_geohashes = list(set(list(train['geohash']) + list(test['geohash'])))
gh_lat, gh_lon = {}, {}
for gh in all_geohashes:
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        cd = base32.index(ch)
        for mask in [16, 8, 4, 2, 1]:
            if even:
                mid = (lon_lo + lon_hi) / 2
                if cd & mask: lon_lo = mid
                else: lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if cd & mask: lat_lo = mid
                else: lat_hi = mid
            even = not even
    gh_lat[gh] = (lat_lo + lat_hi) / 2
    gh_lon[gh] = (lon_lo + lon_hi) / 2

# ── Time & location features ──
for df in [train, test]:
    df['tmin'] = df['timestamp'].str.split(':').str[0].astype(int) * 60 \
               + df['timestamp'].str.split(':').str[1].astype(int)
    df['hour'] = df['tmin'] // 60
    df['tod_bucket'] = df['tmin'] // 30
    df['quarter'] = df['tmin'] // 15  # finer granularity

    # Cyclical time
    df['min_sin']  = np.sin(2 * np.pi * df['tmin'] / 1440)
    df['min_cos']  = np.cos(2 * np.pi * df['tmin'] / 1440)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Location
    df['lat'] = df['geohash'].map(gh_lat)
    df['lon'] = df['geohash'].map(gh_lon)

    # Geohash groupings
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

# ── Missing temp indicator & fill ──
train['temp_missing'] = train['Temperature'].isna().astype(int)
test['temp_missing']  = test['Temperature'].isna().astype(int)

temp_by_gh = train.groupby('geohash')['Temperature'].median()
glob_temp  = train['Temperature'].median()
train['Temperature'] = train['Temperature'].fillna(train['geohash'].map(temp_by_gh)).fillna(glob_temp)
test['Temperature']  = test['Temperature'].fillna(test['geohash'].map(temp_by_gh)).fillna(glob_temp)

# ── Categorical encoding ──
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

# ── Interaction features ──
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

# ── Frequency encoding (no target leakage) ──
gh_freq = train['geohash'].value_counts().to_dict()
for df in [train, test]:
    df['gh_freq'] = df['geohash'].map(gh_freq).fillna(0)

# ── Log target ──
train['y'] = np.log1p(train['demand'])
GM = train['demand'].mean()

# ── Target encoding: OOF for train, full-train for test ──
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
    # Test: use full train
    agg = train.groupby(col)['demand'].agg(['count', 'mean'])
    smooth = (agg['count'] * agg['mean'] + m * GM) / (agg['count'] + m)
    test['te_' + col] = test[col].map(smooth).fillna(GM).values

# ── OOF aggregation features (leak-free) ──
for col_name, agg_func, fill_val, feat_name in [
    ('geohash', 'std', 0, 'te_geohash_std'),
    ('geohash', lambda x: x.quantile(0.5), GM, 'gh_med'),
    ('geohash', lambda x: x.quantile(0.25), GM, 'gh_q25'),
    ('geohash', lambda x: x.quantile(0.75), GM, 'gh_q75'),
]:
    oof = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(col_name)['demand'].agg(agg_func)
        if isinstance(agg, pd.Series):
            agg = agg.fillna(fill_val)
        oof[val_idx] = train.iloc[val_idx][col_name].map(agg).fillna(fill_val).values
    train[feat_name] = oof
    # Test: full train
    agg_full = train.groupby(col_name)['demand'].agg(agg_func)
    if isinstance(agg_full, pd.Series):
        agg_full = agg_full.fillna(fill_val)
    test[feat_name] = test[col_name].map(agg_full).fillna(fill_val)

# OOF gh_hour_med and gh_hour_std
for agg_name, fill_val in [('median', GM), ('std', 0)]:
    feat = 'gh_hour_' + ('med' if agg_name == 'median' else 'std')
    oof = np.zeros(len(train))
    for tr_idx, val_idx in kf.split(train):
        agg = train.iloc[tr_idx].groupby(['geohash', 'hour'])['demand'].agg(agg_name)
        lookup = train.iloc[val_idx].set_index(['geohash', 'hour']).index
        oof[val_idx] = [agg.get(k, fill_val) for k in lookup]
    train[feat] = oof
    # Test: full train
    agg_full = train.groupby(['geohash', 'hour'])['demand'].agg(agg_name)
    lookup_test = test.set_index(['geohash', 'hour']).index
    test[feat] = [agg_full.get(k, fill_val) for k in lookup_test]

# ── Feature list ──
te_feature_names = ['te_' + col for col in te_col_defs.keys()]
agg_feat_names = ['te_geohash_std', 'gh_med', 'gh_q25', 'gh_q75', 'gh_hour_med', 'gh_hour_std']

features = [
    'day', 'tmin', 'hour', 'tod_bucket', 'quarter',
    'min_sin', 'min_cos', 'hour_sin', 'hour_cos',
    'lat', 'lon', 'NumberofLanes', 'Temperature', 'temp_missing',
    'LargeVehicles_e', 'Landmarks_e',
    'is_morning_rush', 'is_evening_rush', 'is_night',
    'gh_freq',
    'temp_x_hour', 'lanes_x_road', 'lanes_x_hour', 'lat_x_lon',
    'temp_x_lanes', 'rush_x_lanes', 'temp_sq', 'lat_x_hour', 'lon_x_hour',
] + road_cols + weather_cols + te_feature_names + agg_feat_names

print(f"Total features: {len(features)}")

# ── 5-Fold Model Training ──
X_all = train[features]
y_all = train['y']

oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_cb  = np.zeros(len(train))
test_xgb = np.zeros(len(test))
test_lgb = np.zeros(len(test))
test_cb  = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n=== Fold {fold} ===")
    X_f_tr, X_f_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_f_tr, y_f_val = y_all.iloc[tr_idx], y_all.iloc[val_idx]

    # XGBoost
    m_xgb = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.015,
        max_depth=8, min_child_weight=4,
        subsample=0.8, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=2.0,
        gamma=0.1, random_state=42, n_jobs=-1,
        early_stopping_rounds=150, eval_metric='rmse'
    )
    m_xgb.fit(X_f_tr, y_f_tr, eval_set=[(X_f_val, y_f_val)], verbose=False)
    oof_xgb[val_idx] = m_xgb.predict(X_f_val)
    test_xgb += m_xgb.predict(test[features]) / 5

    # LightGBM
    m_lgb = lgb.LGBMRegressor(
        n_estimators=5000, learning_rate=0.015,
        max_depth=8, num_leaves=200,
        min_child_samples=20,
        subsample=0.8, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    m_lgb.fit(X_f_tr, y_f_tr, eval_set=[(X_f_val, y_f_val)],
              callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = m_lgb.predict(X_f_val)
    test_lgb += m_lgb.predict(test[features]) / 5

    # CatBoost
    m_cb = CatBoostRegressor(
        iterations=5000, learning_rate=0.015,
        depth=8, l2_leaf_reg=2.0,
        subsample=0.8, random_seed=42, verbose=0,
        early_stopping_rounds=150
    )
    m_cb.fit(X_f_tr, y_f_tr, eval_set=(X_f_val, y_f_val))
    oof_cb[val_idx] = m_cb.predict(X_f_val)
    test_cb += m_cb.predict(test[features]) / 5

    v_xgb = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_xgb[val_idx]), 1e-7, 1))
    v_lgb = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_lgb[val_idx]), 1e-7, 1))
    v_cb  = r2_score(train['demand'].iloc[val_idx], np.clip(np.expm1(oof_cb[val_idx]), 1e-7, 1))
    print(f"  XGB={v_xgb:.4f}  LGB={v_lgb:.4f}  CB={v_cb:.4f}")

# ── OOF scores ──
p_oof_xgb = np.clip(np.expm1(oof_xgb), 1e-7, 1.0)
p_oof_lgb = np.clip(np.expm1(oof_lgb), 1e-7, 1.0)
p_oof_cb  = np.clip(np.expm1(oof_cb), 1e-7, 1.0)
y_true = train['demand'].values

print(f"\n{'='*50}")
print(f"OOF XGB R2: {r2_score(y_true, p_oof_xgb):.4f}")
print(f"OOF LGB R2: {r2_score(y_true, p_oof_lgb):.4f}")
print(f"OOF CB  R2: {r2_score(y_true, p_oof_cb):.4f}")

# ── Find best ensemble weights ──
best_r2 = -1
best_w = None
for w1 in np.arange(0.05, 0.8, 0.05):
    for w2 in np.arange(0.05, 0.8, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05: continue
        p_ens = w1 * p_oof_xgb + w2 * p_oof_lgb + w3 * p_oof_cb
        r2 = r2_score(y_true, p_ens)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (round(w1, 2), round(w2, 2), round(w3, 2))

print(f"\nBest weights: XGB={best_w[0]}, LGB={best_w[1]}, CB={best_w[2]}")
print(f"OOF Ensemble R2: {best_r2:.4f}")

score = max(0, 100 * best_r2)
print(f"\n>>> Competition Score = max(0, 100 * R2) = {score:.2f}")

# ── Generate submission ──
t_xgb = np.clip(np.expm1(test_xgb), 1e-7, 1.0)
t_lgb = np.clip(np.expm1(test_lgb), 1e-7, 1.0)
t_cb  = np.clip(np.expm1(test_cb), 1e-7, 1.0)

test['demand'] = best_w[0] * t_xgb + best_w[1] * t_lgb + best_w[2] * t_cb
submission = test[['Index', 'demand']]
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission shape: {submission.shape}")
print(f"Columns: {list(submission.columns)}")
print(f"Demand range: [{submission['demand'].min():.6f}, {submission['demand'].max():.6f}]")
print("Submission saved to submission.csv")
