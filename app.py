import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.preprocessing import LabelEncoder

st.title("🐎【完全版】スマホで育てる！競馬AIマスターアプリ")
st.write("過去データ学習・結果追加・テキスト一発ペースト予測のすべてがここに！✨🔥")

def load_model():
    try:
        if os.path.exists('keiba_ai_model.pkl'):
            return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None
    return None

model = load_model()

# 分割されたマスターデータの自動合体
df_m_auto = pd.DataFrame()
dfs = []
for filename in ['keiba_master_data_part1.csv', 'keiba_master_data_part2.csv', 'keiba_master_data.csv']:
    if os.path.exists(filename):
        try:
            temp_df = pd.read_csv(filename, encoding='cp932', low_memory=False)
        except Exception:
            temp_df = pd.read_csv(filename, encoding='utf-8-sig', low_memory=False)
        dfs.append(temp_df)

if dfs:
    df_m_auto = pd.concat(dfs, ignore_index=True)
    df_m_auto.columns = [str(c).strip() for c in df_m_auto.columns]
    for col_candidate in ['着順', '順位', 'Rank', 'RANK']:
        if col_candidate in df_m_auto.columns and 'rank' not in df_m_auto.columns:
            df_m_auto['rank'] = df_m_auto[col_candidate]

def parse_corner_position(val):
    try:
        s = str(val).strip()
        if '-' in s:
            parts = s.split('-')
            return float(parts[-1])
        elif s.isdigit():
            return float(s)
    except:
        pass
    return 8.0

if not df_m_auto.empty and 'corner' in df_m_auto.columns:
    df_m_auto['corner_4th'] = df_m_auto['corner'].apply(parse_corner_position)
else:
    df_m_auto['corner_4th'] = 8.0

def parse_time_to_sec(val):
    try:
        s = str(val).strip()
        if ':' in s:
            m, rest = s.split(':')
            return float(m) * 60 + float(rest)
        elif s and s != 'nan':
            return float(s)
    except:
        pass
    return 0.0

if not df_m_auto.empty and 'time' in df_m_auto.columns:
    df_m_auto['time_sec'] = df_m_auto['time'].apply(parse_time_to_sec)
else:
    df_m_auto['time_sec'] = 0.0

jockey_win_rates = {}
if not df_m_auto.empty and 'jockey' in df_m_auto.columns and 'rank' in df_m_auto.columns:
    df_m_auto['rank'] = pd.to_numeric(df_m_auto['rank'], errors='coerce')
    jockey_stats = df_m_auto.groupby('jockey').agg(
        total=('rank', 'count'),
        wins=('rank', lambda x: (x == 1).sum())
    )
    for jock, row in jockey_stats.iterrows():
        if row['total'] > 0:
            jockey_win_rates[jock] = row['wins'] / row['total']

tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果を追加", "🧠 AI再学習"])

with tab1:
    st.subheader("🚀 勝ち馬のガチ予測（スマホ対応テキスト一発ペースト）")
   
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        p_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="p_place")
        p_class = st.selectbox("クラス", ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"], key="p_class")
    with col_p2:
        p_track = st.selectbox("トラック", ["芝", "ダート", "障害"], key="p_track")
        p_distance = st.number_input("距離 (m)", value=2000, step=100, key="p_distance")
    with col_p3:
        p_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="p_condition")

    st.markdown("---")
    st.markdown("### 📋 出馬表テキストの一発ペースト")
    raw_text = st.text_area("ここにnetkeibaの出馬表をペースト", height=150, key="raw_text_input")

    input_data_list = []
    if raw_text.strip():
        lines = raw_text.strip().split('\n')
        current_horse = None
        cleaned_lines = [line.strip() for line in lines if line.strip() != ""]
       
        i = 0
        while i < len(cleaned_lines):
            line = cleaned_lines[i]
            if line.isdigit() and 1 <= int(line) <= 18:
                if current_horse:
                    input_data_list.append(current_horse)
               
                umaban = int(line)
                waku = min(8, max(1, ((umaban - 1) // 2) + 1))
                h_name = f"馬番{umaban}"
                odds = 10.0
                popularity = umaban
                jockey = "不明"
                weight = 56.0
                age = 4
               
                for j in range(i+1, min(i+8, len(cleaned_lines))):
                    sub_line = cleaned_lines[j]
                    if "データベース" in sub_line:
                        h_name = sub_line.replace("のデータベース", "").strip()
                    elif any(s in sub_line for s in ["牡", "牝", "セ"]) and len(sub_line) <= 6:
                        for char in sub_line:
                            if char.isdigit():
                                age = int(char)
                    elif "人気" in sub_line:
                        try:
                            pop_val = int(sub_line.replace("人気", "").strip())
                            popularity = pop_val
                        except:
                            pass
                    try:
                        val = float(sub_line)
                        if 0.1 <= val < 2000:
                            if val == int(val) and val <= 18 and popularity == umaban:
                                popularity = int(val)
                            elif val >= 1.0:
                                odds = val
                    except ValueError:
                        if len(sub_line) >= 2 and not any(c.isdigit() for c in sub_line) and "人気" not in sub_line and "厩舎" not in sub_line:
                            if jockey == "不明":
                                jockey = sub_line

                current_horse = {
                    'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                    'race_class': p_class, 'waku': waku, 'umaban': umaban, 'name': h_name, 'sex': '牡', 'age': age, 'sire': '不明',
                    'odds': odds, 'popularity': popularity, 'weight': weight, 'jockey': jockey,
                    'jockey_win_rate': jockey_win_rates.get(jockey, 0.08), 'blinker': 0, 'corner_4th': 8.0, 'time_sec': 0.0
                }
            i += 1
       
        if current_horse:
            input_data_list.append(current_horse)

    if len(input_data_list) == 0:
        st.warning("⚠️ テキスト未入力のため、手動モードで8頭分のデフォルトを表示しています。")
        for i in range(8):
            input_data_list.append({
                'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                'race_class': p_class, 'waku': 1, 'umaban': i+1, 'name': f"馬番{i+1}", 'sex': '牡', 'age': 4, 'sire': '不明',
                'odds': 10.0, 'popularity': i+1, 'weight': 56.0, 'jockey': '不明', 'jockey_win_rate': 0.08, 'blinker': 0, 'corner_4th': 8.0, 'time_sec': 0.0
            })
    else:
        st.success(f"✨ テキストから出走馬 **{len(input_data_list)}頭** を正確に検出しました！")

    current_model = load_model()
    if current_model is None:
        st.error("⚠️ AIモデルが見つかりません。「AI再学習」タブからモデルを作成してください。")
    elif st.button("🚀 ガチ予測を実行する！"):
        df_input = pd.DataFrame(input_data_list)
        df_full = pd.concat([df_m_auto, df_input], ignore_index=True) if not df_m_auto.empty else df_input
        df_full = df_full.loc[:, ~df_full.columns.duplicated()]

        for col in ['place', 'track', 'condition', 'sire', 'race_class']:
            if col in df_full.columns:
                le = LabelEncoder()
                df_full[col] = le.fit_transform(df_full[col].astype(str))

        df_input_encoded = df_full.tail(len(df_input)).copy()
        features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker', 'corner_4th', 'race_class', 'time_sec']
        for feat in features:
            if feat not in df_input_encoded.columns:
                df_input_encoded[feat] = 0

        X_pred = df_input_encoded[features].fillna(0)
        try:
            st.balloons()
            st.subheader("🎯 ガチAI予測結果ランキング")
            probs = current_model.predict_proba(X_pred)[:, 1]
            df_input['win_prob'] = probs * 100
            df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
            for idx, row in df_input.iterrows():
                st.write(f"**第 {idx+1} 位**: 馬番 {row['umaban']} 🐴 **{row['name']}** (予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍 / 騎手: {row['jockey']})")
        except Exception as e:
            st.warning(f"⚠️ エラー: {e}")

with tab2:
    st.subheader("📝 レース結果をマスターに追加する")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        r_year = st.number_input("年", min_value=2000, max_value=2030, value=2026, key="r_year")
    with col_d2:
        r_month = st.number_input("月", min_value=1, max_value=12, value=6, key="r_month")
    with col_d3:
        r_day = st.number_input("日", min_value=1, max_value=31, value=1, key="r_day")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
        race_class = st.selectbox("クラス", ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"], key="r_class")
    with col_r2:
        track_type = st.selectbox("トラック", ["芝", "ダート", "障害"], key="r_track")
        distance = st.number_input("距離 (m)", value=2000, step=100, key="r_distance")
    with col_r3:
        condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="r_cond")

    res_num_horses = st.slider("出走頭数", min_value=1, max_value=18, value=8, key="res_num")
    new_data_list = []
    for i in range(res_num_horses):
        u_num = i + 1
        with st.expander(f"馬番 {u_num} の結果入力", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                r_name = st.text_input("馬名", f"馬番{u_num}", key=f"r_name_{i}")
                r_rank = st.number_input("確定着順", min_value=1, max_value=18, value=1, key=f"r_rank_{i}")
                r_odds = st.number_input("単勝オッズ", value=10.0, min_value=0.0, step=0.1, key=f"r_odds_{i}")
                r_pop = st.number_input("人気順", value=u_num, min_value=1, step=1, key=f"r_pop_{i}")
                r_blinker_str = st.selectbox("ブリンカー", ["なし", "B (あり)"], key=f"r_blinker_{i}")
                r_blinker = 1 if "B" in r_blinker_str else 0

            with col_b:
                r_jockey = st.text_input("騎手名", "不明", key=f"r_jockey_{i}")
                r_weight = st.number_input("斤量", value=56.0, step=0.5, key=f"r_weight_{i}")
                r_sire = st.text_input("父馬名", "不明", key=f"r_sire_{i}")
                r_age = st.number_input("年齢", min_value=2, max_value=15, value=4, key=f"r_age_{i}")
                r_corner = st.text_input("通過順 (例: 1-1-1-1)", "5-5-4-3", key=f"r_corner_{i}")
                r_time = st.text_input("走破タイム (例: 1:45.2)", "2:00.0", key=f"r_time_{i}")

            new_data_list.append({
                'year': r_year, 'month': r_month, 'day': r_day,
                'place': race_place, 'track': track_type, 'distance': distance, 'condition': condition,
                'race_class': race_class, 'waku': ((u_num-1)//2)+1, 'umaban': u_num, 'name': r_name,
                'sex': '牡', 'age': r_age, 'jockey': r_jockey, 'sire': r_sire, 'weight': r_weight,
                'rank': r_rank, 'odds': r_odds, 'popularity': r_pop, 'blinker': r_blinker,
                'corner': r_corner, 'corner_4th': parse_corner_position(r_corner),
                'time': r_time, 'time_sec': parse_time_to_sec(r_time)
            })

    if st.button("🚀 追加データをマスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        df_combined = pd.concat([df_m_auto, df_new], ignore_index=True) if not df_m_auto.empty else df_new
        df_combined.to_csv('keiba_master_data_part1.csv', index=False, encoding='cp932')
        st.balloons()
        st.success("🎉 日時・クラス・走破タイムを含めてマスターに保存されました！")

with tab3:
    st.subheader("🧠 ガチAIを再学習させる")
    if not df_m_auto.empty:
        st.success(f"📂 マスターデータ行数: **{len(df_m_auto):,} 行** (正常に読み込み成功！)")
        if st.button("🚀 フルデータでAIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
                df_train = df_m_auto.copy().loc[:, ~df_m_auto.columns.duplicated()]
               
                if 'rank' not in df_train.columns:
                    for col in df_train.columns:
                        if '順' in str(col) or '着' in str(col):
                            df_train['rank'] = df_train[col]
                            break
                    if 'rank' not in df_train.columns:
                        df_train['rank'] = 1
               
                if 'corner' in df_train.columns and 'corner_4th' not in df_train.columns:
                    df_train['corner_4th'] = df_train['corner'].apply(parse_corner_position)
                elif 'corner_4th' not in df_train.columns:
                    df_train['corner_4th'] = 8.0

                if 'time' in df_train.columns and 'time_sec' not in df_train.columns:
                    df_train['time_sec'] = df_train['time'].apply(parse_time_to_sec)
                elif 'time_sec' not in df_train.columns:
                    df_train['time_sec'] = 0.0

                if 'blinker' not in df_train.columns:
                    df_train['blinker'] = 0

                if len(df_train) > 50000: df_train = df_train.sample(n=50000, random_state=42)
               
                df_train['target'] = (pd.to_numeric(df_train['rank'], errors='coerce') == 1).astype(int)
               
                if 'jockey' in df_train.columns and 'rank' in df_train.columns:
                    jockey_stats_train = df_train.groupby('jockey').agg(
                        total=('rank', 'count'),
                        wins=('rank', lambda x: (x == 1).sum())
                    )
                    train_rates = {j: row['wins'] / row['total'] for j, row in jockey_stats_train.iterrows() if row['total'] > 0}
                    df_train['jockey_win_rate'] = df_train['jockey'].map(train_rates).fillna(0.08)
                else:
                    df_train['jockey_win_rate'] = 0.08

                for col in ['place', 'track', 'condition', 'sire', 'race_class']:
                    if col in df_train.columns:
                        df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))
               
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker', 'corner_4th', 'race_class', 'time_sec']
                for f in features:
                    if f not in df_train.columns: df_train[f] = 0
               
                X = df_train[features].fillna(0)
                y = df_train['target']
                clf = lgb.LGBMClassifier(random_state=42)
                clf.fit(X, y)
               
                joblib.dump(clf, 'keiba_ai_model.pkl')
               
                st.balloons()
                st.success("🎉 再学習完了！モデルが正常にアップデートされました！")
            except Exception as e:
                st.warning(f"⚠️ 学習エラー: {e}")
    else:
        st.warning("⚠️ マスターデータが見つかりません。ファイル名を確認してください。")
