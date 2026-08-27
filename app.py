import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import re
from sklearn.preprocessing import LabelEncoder

st.title("🐎【ガチ仕様】スマホで育てる！競馬AIマスターアプリ（コピペ対応版）")

@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# --- 共通：大容量データの確実読込＆重複のない列名自動マッピング ---
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
            pass

if len(dfs) > 0:
    df_m_auto = pd.concat(dfs, ignore_index=True)
   
    rename_cols = {}
    existing_cols = set(df_m_auto.columns)
   
    for col in df_m_auto.columns:
        c_str = str(col).strip()
        target_name = None
       
        if any(w in c_str for w in ['着順', '確定着順']):
            target_name = 'rank'
        elif any(w in c_str for w in ['騎手', '騎手名']):
            target_name = 'jockey'
        elif any(w in c_str for w in ['父', '血統(父)', '種牡馬']):
            target_name = 'sire'
        elif any(w in c_str for w in ['馬名']):
            target_name = 'name'
        elif any(w in c_str for w in ['オッズ', '単勝']):
            target_name = 'odds'
        elif any(w in c_str for w in ['人気', '順位']):
            target_name = 'popularity'
        elif any(w in c_str for w in ['斤量', '負担重量']):
            target_name = 'weight'
        elif any(w in c_str for w in ['年齢', '馬齢']):
            target_name = 'age'
        elif any(w in c_str for w in ['枠番', '枠']):
            target_name = 'waku'
        elif any(w in c_str for w in ['馬番']):
            target_name = 'umaban'
        elif any(w in c_str for w in ['距離']):
            target_name = 'distance'
        elif any(w in c_str for w in ['開催', '場所', '競馬場']):
            target_name = 'place'
        elif any(w in c_str for w in ['芝・ダート', 'トラック', 'コース']):
            target_name = 'track'
        elif any(w in c_str for w in ['馬場', '馬場状態']):
            target_name = 'condition'
           
        if target_name and target_name not in existing_cols and target_name not in rename_cols.values():
            if col != target_name:
                rename_cols[col] = target_name
   
    if rename_cols:
        df_m_auto = df_m_auto.rename(columns=rename_cols)

    df_m_auto = df_m_auto.loc[:, ~df_m_auto.columns.duplicated()]

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
tab1, tab2, tab3, tab4 = st.tabs(["🚀 ガチ予測（コピペ対応）", "📝 結果入力", "🧠 AI再学習", "⚙️ 従来の個別入力"])

with tab1:
    st.subheader("🚀 ネット競馬の出馬表コピペで一括予測！")
    st.info("💡 ネット競馬などの出馬表をコピーして下のボックスに貼り付けるだけで、全頭の予測ができます！")
   
    raw_text = st.text_area("ここに出馬表を貼り付け", height=150, placeholder="例: 1 1 コントレイル 牡3 56.0 福永祐一 1.8 ...")
   
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        p_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="p_place")
    with col_p2:
        p_track = st.selectbox("トラック", ["芝", "ダート", "障害"], key="p_track")
        p_distance = st.number_input("距離 (m)", value=1600, step=100, key="p_distance")
    with col_p3:
        p_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="p_condition")

    if st.button("🚀 貼り付けデータで一括予測を実行！"):
        if not raw_text.strip():
            st.warning("⚠️ テキストが入力されていません。")
        else:
            lines = raw_text.split("\n")
            parsed_data = []
           
            for line in lines:
                numbers = re.findall(r'\d+\.\d+|\d+', line)
                words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FA5A-Za-z]+', line)
               
                if words:
                    horse_name = words[0]
                    odds = float(numbers[-1]) if numbers and '.' in numbers[-1] else 10.0
                    wakuban = int(numbers[0]) if numbers else 1
                    umaban = int(numbers[1]) if len(numbers) > 1 else 1
                   
                    jock_rate = jockey_win_rates.get("不明", 0.08)
                   
                    parsed_data.append({
                        'place': p_place,
                        'track': p_track,
                        'distance': p_distance,
                        'condition': p_condition,
                        'waku': wakuban,
                        'umaban': umaban,
                        'name': horse_name,
                        'sex': '牡',
                        'age': 3,
                        'sire': '不明',
                        'odds': odds,
                        'popularity': umaban,
                        'weight': 56.0,
                        'jockey': '不明',
                        'jockey_win_rate': jock_rate,
                        'blinker': 0
                    })
           
            if parsed_data and model is not None:
                df_input = pd.DataFrame(parsed_data)
                df_full = pd.concat([df_m_auto, df_input], ignore_index=True) if len(df_m_auto) > 0 else df_input
                df_full = df_full.loc[:, ~df_full.columns.duplicated()]

                cat_cols = ['place', 'track', 'condition', 'sire']
                for col in cat_cols:
                    if col in df_full.columns:
                        le = LabelEncoder()
                        df_full[col] = df_full[col].astype(str)
                        df_full[col] = le.fit_transform(df_full[col])

                df_input_encoded = df_full.tail(len(df_input)).copy()
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
                X_pred = df_input_encoded[features].fillna(0)

                probs = model.predict_proba(X_pred)[:, 1]
                df_input['win_prob'] = probs * 100
                df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

                st.balloons()
                st.subheader("🎯 一括予測ランキング結果")
                for idx, row in df_input.iterrows():
                    st.write(f"**第 {idx+1} 位**: 馬番 {row['umaban']} 🐴 **{row['name']}** (予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍)")
            else:
                st.error("⚠️ データをうまく読み込めなかったか、AIモデルがありません。")

with tab2:
    st.subheader("📝 レース結果をマスターに追加する")
    race_date = st.date_input("開催日", datetime.date(2026, 6, 1))
    race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
    race_number = st.number_input("レース番号 (R)", min_value=1, max_value=12, value=11, key="r_num")
    race_name = st.text_input("レース名", "", placeholder="例: 日本ダービー", key="r_name_input")
    track_type = st.selectbox("トラック", ["芝", "ダート", "障害"], key="r_track")
    distance = st.number_input("距離 (m)", value=1600, step=100, key="r_distance")
    condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="r_cond")
    weather = st.selectbox("天気", ["晴", "曇", "雨", "小雨", "雪"], key="r_weather")

    res_num_horses = st.slider("出走頭数", min_value=1, max_value=18, value=8, key="res_num")
    new_data_list = []
    for i in range(res_num_horses):
        auto_umaban = i + 1
        auto_waku = ((auto_umaban - 1) // 2) + 1
        with st.expander(f"【結果入力】馬番 {auto_umaban}", expanded=False):
            r_name = st.text_input(f"馬名", "", key=f"r_name_{i}")
            r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=1, key=f"r_rank_{i}")
            r_odds = st.number_input(f"単勝オッズ", value=1.0, min_value=0.0, step=0.1, key=f"r_odds_{i}")
            new_data_list.append({
                'date': race_date, 'place': race_place, 'race_number': race_number, 'race_name': race_name,
                'track': track_type, 'distance': distance, 'condition': condition, 'weather': weather,
                'waku': auto_waku, 'umaban': auto_umaban, 'horse_name': r_name if r_name else f"馬番{auto_umaban}",
                'sex': '牡', 'age': 3, 'jockey': '不明', 'trainer': '不明', 'stable': '栗東',
                'sire': '不明', 'dam': '不明', 'weight': 56.0, 'blinker': '', 'rank': r_rank,
                'time': '0:00.0', 'corner': '0', 'odds': r_odds, 'popularity': 1
            })

    if st.button("🚀 追加データをマスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        MASTER_FILENAME = 'keiba_master_data.csv'
        if os.path.exists(MASTER_FILENAME):
            try:
                df_master = pd.read_csv(MASTER_FILENAME, encoding='cp932', low_memory=False)
            except Exception:
                df_master = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig', low_memory=False)
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(MASTER_FILENAME, index=False, encoding='cp932')
        st.success("🎉 追加成功！マスターデータに保存されました！")

with tab3:
    st.subheader("🧠 全データでガチAIを再学習させる")
    if len(df_m_auto) > 10:
        if st.button("🚀 フルデータでAIを再学習・アップデートする！"):
            import lightgbm as lgb
            target_train_df = df_m_auto.copy()
            if len(target_train_df) > 50000:
                target_train_df = target_train_df.sample(n=50000, random_state=42)
           
            target_train_df['jockey_win_rate'] = 0.08
            target_train_df['blinker'] = 0
            cat_cols = ['place', 'track', 'condition', 'sire']
            for col in cat_cols:
                if col in target_train_df.columns:
                    le = LabelEncoder()
                    target_train_df[col] = le.fit_transform(target_train_df[col].astype(str))
                else:
                    target_train_df[col] = 0

            target_train_df['target'] = (target_train_df['rank'] == 1).astype(int)
            features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
            X = target_train_df[features].fillna(0)
            y = target_train_df['target']
           
            clf = lgb.LGBMClassifier(random_state=42)
            clf.fit(X, y)
            joblib.dump(clf, 'keiba_ai_model.pkl')
            st.success("🎉 再学習が完了しました！")

with tab4:
    st.subheader("⚙️ 従来の個別入力モード")
    user_odds = st.number_input("単勝オッズ微調整", value=5.0)
    user_umaban = st.number_input("馬番", value=1)
    if st.button("個別予測実行"):
        st.write("個別入力での予測検証用タブです。")
