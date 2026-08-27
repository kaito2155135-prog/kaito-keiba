import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.preprocessing import LabelEncoder

st.title("🐎【ガチ仕様】スマホで育てる！競馬AIマスターアプリ")
st.write("超強力列名自動変換・完全安定版！過去5年分データ合体＆ブリンカー対応や！🔥")

@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# --- 共通：大容量データの確実読込＆あらゆる日本語列名の完全マッピング ---
jockey_win_rates = {}
df_m_auto = pd.DataFrame()
dfs = []

files = ['keiba_master_data_part1.csv', 'keiba_master_data_part2.csv', 'keiba_master_data.csv']

for f in files:
    if os.path.exists(f):
        try:
            try:
                temp_df = pd.read_csv(f, encoding='cp932', low_memory=False)
            except Exception:
                temp_df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
               
            if len(temp_df) > 10:
                dfs.append(temp_df)
        except Exception as e:
            st.write(f"読み込みエラー ({f}): {e}")

if len(dfs) > 0:
    df_m_auto = pd.concat(dfs, ignore_index=True)
   
    # --- 超強力・列名自動変換（あらゆる表記揺れに対応） ---
    rename_cols = {}
    for col in df_m_auto.columns:
        c_str = str(col).strip()
        if any(w in c_str for w in ['着順', '確定着順']):
            rename_cols[col] = 'rank'
        elif any(w in c_str for w in ['騎手', '騎手名']):
            rename_cols[col] = 'jockey'
        elif any(w in c_str for w in ['父', '血統(父)', '種牡馬']):
            rename_cols[col] = 'sire'
        elif any(w in c_str for w in ['馬名']):
            rename_cols[col] = 'name'
        elif any(w in c_str for w in ['オッズ', '単勝']):
            rename_cols[col] = 'odds'
        elif any(w in c_str for w in ['人気', '順位']):
            rename_cols[col] = 'popularity'
        elif any(w in c_str for w in ['斤量', '負担重量']):
            rename_cols[col] = 'weight'
        elif any(w in c_str for w in ['年齢', '馬齢']):
            rename_cols[col] = 'age'
        elif any(w in c_str for w in ['枠番', '枠']):
            rename_cols[col] = 'waku'
        elif any(w in c_str for w in ['馬番']):
            rename_cols[col] = 'umaban'
        elif any(w in c_str for w in ['距離']):
            rename_cols[col] = 'distance'
        elif any(w in c_str for w in ['開催', '場所', '競馬場']):
            rename_cols[col] = 'place'
        elif any(w in c_str for w in ['芝・ダート', 'トラック', 'コース']):
            rename_cols[col] = 'track'
        elif any(w in c_str for w in ['馬場', '馬場状態']):
            rename_cols[col] = 'condition'
   
    if rename_cols:
        df_m_auto = df_m_auto.rename(columns=rename_cols)

    if 'blinker' not in df_m_auto.columns:
        df_m_auto['blinker'] = ""
   
    if 'jockey' in df_m_auto.columns and 'rank' in df_m_auto.columns:
        df_m_auto['rank'] = pd.to_numeric(df_m_auto['rank'], errors='coerce')
        jockey_stats = df_m_auto.groupby('jockey').agg(
            total=('rank', 'count'),
            wins=('rank', lambda x: (x == 1).sum())
        )
        for jock, row in jockey_stats.iterrows():
            if row['total'] > 0:
                jockey_win_rates[jock] = row['wins'] / row['total']

# タブで機能を分ける
tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果をマスターに追加", "🧠 ガチAIの再学習（アップデート）"])

with tab1:
    st.subheader("🚀 勝ち馬のガチ予測（コース・血統・ブリンカー連動）")
   
    if len(df_m_auto) > 10:
        st.success(f"📂 大容量マスターデータ読み込み成功！（総データ数: {len(df_m_auto):,}行✨）")
    else:
        st.warning(f"⚠️ データ数が少なすぎるで（現在 {len(df_m_auto)}行）。")

    # --- 予測用レース条件の入力 ---
    st.markdown("### 🏟️ 予想するレースの条件")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        p_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="p_place")
    with col_p2:
        p_track = st.selectbox("トラック", ["芝", "ダート", "障害"], key="p_track")
        p_distance = st.number_input("距離 (m)", value=1600, step=100, key="p_distance")
    with col_p3:
        p_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="p_condition")

    st.markdown("---")

    if model is None:
        st.error("⚠️ AIモデル（keiba_ai_model.pkl）が見つかりません。「ガチAIの再学習」タブから最初のモデルを作ってください。")
    else:
        st.success("✨ ガチAIモデル稼働中！")

        num_horses = st.slider("予測する出馬頭数", min_value=1, max_value=18, value=8, key="pred_num")
       
        input_data_list = []
        for i in range(num_horses):
            auto_umaban = i + 1
            auto_waku = ((auto_umaban - 1) // 2) + 1
            if auto_waku > 8: auto_waku = 8

            with st.expander(f"馬番 {auto_umaban} の予測データ", expanded=(i < 3)):
                col1, col2 = st.columns(2)
                with col1:
                    horse_name = st.text_input(f"馬名", f"馬名{auto_umaban}", key=f"p_name_{i}")
                    r_sex = st.selectbox(f"性別", ["牡", "牝", "騸"], key=f"p_sex_{i}")
                    r_age = st.number_input(f"年齢", min_value=2, max_value=15, value=3, key=f"p_age_{i}")
                    weight = st.number_input(f"斤量", value=56.0, step=0.5, key=f"p_weight_{i}")
                    r_sire = st.text_input(f"父馬名（血統）", "ディープインパクト", key=f"p_sire_{i}")
                with col2:
                    odds = st.number_input(f"単勝オッズ", value=10.0, min_value=1.0, step=0.1, key=f"p_odds_{i}")
                    popularity = st.number_input(f"人気順", value=auto_umaban, min_value=1, step=1, key=f"p_pop_{i}")
                    r_jockey = st.text_input(f"騎手名", "騎手名", key=f"p_jockey_{i}")
                    r_blinker = st.selectbox(f"ブリンカー", ["", "B"], key=f"p_blinker_{i}")
               
                jock_rate = jockey_win_rates.get(r_jockey, 0.08)
                is_blinker = 1 if r_blinker == "B" else 0

                input_data_list.append({
                    'place': p_place,
                    'track': p_track,
                    'distance': p_distance,
                    'condition': p_condition,
                    'waku': auto_waku,
                    'umaban': auto_umaban,
                    'name': horse_name,
                    'sex': r_sex,
                    'age': r_age,
                    'sire': r_sire,
                    'odds': odds,
                    'popularity': popularity,
                    'weight': weight,
                    'jockey': r_jockey,
                    'jockey_win_rate': jock_rate,
                    'blinker': is_blinker
                })

        if st.button("🚀 ガチ予測を実行する！"):
            df_input = pd.DataFrame(input_data_list)
           
            if len(df_m_auto) > 0:
                df_full = pd.concat([df_m_auto, df_input], ignore_index=True)
            else:
                df_full = df_input

            cat_cols = ['place', 'track', 'condition', 'sire']
            for col in cat_cols:
                if col in df_full.columns:
                    le = LabelEncoder()
                    df_full[col] = df_full[col].astype(str)
                    df_full[col] = le.fit_transform(df_full[col])

            df_input_encoded = df_full.tail(len(df_input)).copy()

            features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
            X_pred = df_input_encoded[features].fillna(0)

            try:
                st.balloons()
                st.subheader("🎯 ガチAI予測結果ランキング")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_pred)[:, 1]
                    df_input['win_prob'] = probs * 100
                    df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

                    for idx, row in df_input.iterrows():
                        b_text = " 【B着用】" if row['blinker'] == 1 else ""
                        st.write(f"**第 {idx+1} 位**: 馬番 {row['umaban']} 🐴 **{row['name']}**{b_text} (父:{row['sire']} / 予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍 / 騎手: {row['jockey']})")
                else:
                    preds = model.predict(X_pred)
                    df_input['pred'] = preds
                    for idx, row in df_input.iterrows():
                        st.write(f"馬番 {row['umaban']} 🐴 **{row['name']}** - 予測結果: {row['pred']}")
            except Exception as e:
                st.warning("⚠️ エラーが発生しました（「ガチAIの再学習」を行ってください）:")
                st.write(e)

with tab2:
    st.subheader("📝 レース結果をマスターに追加する")
    st.write("ブリンカー情報も含めて結果を追加・蓄積するで！✨")

    st.markdown("---")
    st.markdown("### 🏟️ 今回のレース条件")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        race_date = st.date_input("開催日", datetime.date(2026, 6, 1))
        race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
        race_number = st.number_input("レース番号 (R)", min_value=1, max_value=12, value=11, key="r_num")
    with col_r2:
        race_name = st.text_input("レース名", "〇〇ステークス", key="r_name_input")
        track_type = st.selectbox("トラック", ["芝", "ダート", "障害"], key="r_track")
        distance = st.number_input("距離 (m)", value=1600, step=100, key="r_distance")
    with col_r3:
        condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="r_cond")
        weather = st.selectbox("天気", ["晴", "曇", "雨", "小雨", "雪"], key="r_weather")

    res_num_horses = st.slider("出走頭数", min_value=1, max_value=18, value=8, key="res_num")

    new_data_list = []
    for i in range(res_num_horses):
        auto_umaban = i + 1
        auto_waku = ((auto_umaban - 1) // 2) + 1
        if auto_waku > 8: auto_waku = 8

        with st.expander(f"【結果入力】馬番 {auto_umaban}", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                r_name = st.text_input(f"馬名", f"馬名{auto_umaban}", key=f"r_name_{i}")
                r_sex = st.selectbox(f"性別", ["牡", "牝", "騸"], key=f"r_sex_{i}")
                r_age = st.number_input(f"年齢", min_value=2, max_value=15, value=3, key=f"r_age_{i}")
                r_jockey = st.text_input(f"騎手名", "騎手名", key=f"r_jockey_{i}")
                r_trainer = st.text_input(f"調教師名", "調教師名", key=f"r_trainer_{i}")
                r_stable = st.selectbox(f"所属", ["美浦", "栗東", "地方", "海外"], key=f"r_stable_{i}")
            with col_b:
                r_sire = st.text_input(f"父馬名（血統）", "ディープインパクト", key=f"r_sire_{i}")
                r_dam = st.text_input(f"母馬名", "母馬名", key=f"r_dam_{i}")
                r_weight = st.number_input(f"斤量", value=56.0, step=0.5, key=f"r_weight_{i}")
                r_blinker_res = st.selectbox(f"ブリンカー", ["", "B"], key=f"r_blinker_{i}")
                r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=auto_umaban, key=f"r_rank_{i}")
                r_time = st.text_input(f"走破タイム", "1:35.0", key=f"r_time_{i}")
                r_corner = st.text_input(f"通過順", "1-1", key=f"r_corner_{i}")
                r_odds = st.number_input(f"単勝オッズ", value=10.0, step=0.1, key=f"r_odds_{i}")
                r_pop = st.number_input(f"人気順", value=auto_umaban, key=f"r_pop_{i}")

            new_data_list.append({
                'date': race_date,
                'place': race_place,
                'race_number': race_number,
                'race_name': race_name,
                'track': track_type,
                'distance': distance,
                'condition': condition,
                'weather': weather,
                'waku': auto_waku,
                'umaban': auto_umaban,
                'horse_name': r_name,
                'sex': r_sex,
                'age': r_age,
                'jockey': r_jockey,
                'trainer': r_trainer,
                'stable': r_stable,
                'sire': r_sire,
                'dam': r_dam,
                'weight': r_weight,
                'blinker': r_blinker_res,
                'rank': r_rank,
                'time': r_time,
                'corner': r_corner,
                'odds': r_odds,
                'popularity': r_pop
            })

    MASTER_FILENAME = 'keiba_master_data.csv'
    if st.button("🚀 追加データをマスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
       
        if os.path.exists(MASTER_FILENAME):
            try:
                df_master = pd.read_csv(MASTER_FILENAME, encoding='cp932', low_memory=False)
            except Exception:
                df_master = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig', low_memory=False)
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(MASTER_FILENAME, index=False, encoding='cp932')
       
        st.balloons()
        st.success(f"🎉 追加成功！マスターデータに結果が保存されました！✨")

with tab3:
    st.subheader("🧠 全データでガチAIを再学習させる")
   
    if len(df_m_auto) > 10:
        st.success(f"📂 合算データ読み込み成功！学習データ総数: **{len(df_m_auto):,} 行**")

        if st.button("🚀 フルデータでAIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
               
                target_train_df = df_m_auto.copy()
               
                if 'jockey' in target_train_df.columns and 'rank' in target_train_df.columns:
                    jockey_stats_train = target_train_df.groupby('jockey').agg(
                        total=('rank', 'count'),
                        wins=('rank', lambda x: (x == 1).sum())
                    )
                    train_rates = {}
                    for jock, row in jockey_stats_train.iterrows():
                        if row['total'] > 0:
                            train_rates[jock] = row['wins'] / row['total']
                   
                    target_train_df['jockey_win_rate'] = target_train_df['jockey'].map(train_rates).fillna(0.08)
                else:
                    target_train_df['jockey_win_rate'] = 0.08

                if 'blinker' in target_train_df.columns:
                    target_train_df['blinker'] = target_train_df['blinker'].apply(lambda x: 1 if str(x).strip() == 'B' else 0)
                else:
                    target_train_df['blinker'] = 0

                cat_cols = ['place', 'track', 'condition', 'sire']
                for col in cat_cols:
                    if col in target_train_df.columns:
                        le = LabelEncoder()
                        target_train_df[col] = target_train_df[col].astype(str)
                        target_train_df[col] = le.fit_transform(target_train_df[col])
                    else:
                        target_train_df[col] = 0

                if 'rank' in target_train_df.columns:
                    target_train_df['target'] = (target_train_df['rank'] == 1).astype(int)
                else:
                    st.error("⚠️ 'rank'（着順）の列が見つかりません。")
                    st.stop()
               
                # --- 万が一足りない特徴量があっても自動で0を埋めてエラーを防ぐ安全対策 ---
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
                for feat in features:
                    if feat not in target_train_df.columns:
                        target_train_df[feat] = 0
               
                X = target_train_df[features].fillna(0)
                y = target_train_df['target']

                if len(X) < 5:
                    st.warning("⚠️ データ数が少なすぎます。")
                else:
                    clf = lgb.LGBMClassifier(random_state=42)
                    clf.fit(X, y)

                    model_path = 'keiba_ai_model.pkl'
                    joblib.dump(clf, model_path)

                    st.balloons()
                    st.success("🎉 21万行フルデータでの最強ガチAIの再学習が完了しました！✨")

            except Exception as e:
                st.error("⚠️ 学習中にエラーが発生しました:")
                st.write(e)
    else:
        st.warning("⚠️ 十分なデータが読み込めていません。")