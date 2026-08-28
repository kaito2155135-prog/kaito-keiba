import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.preprocessing import LabelEncoder

st.title("🐎【完全版】スマホで育てる！競馬AIマスターアプリ")
st.write("過去データ学習・結果追加・テキスト一発ペースト予測のすべてがここに！✨🔥")

@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# マスターデータの自動読み込み
df_m_auto = pd.DataFrame()
if os.path.exists('keiba_master_data.csv'):
    try:
        df_m_auto = pd.read_csv('keiba_master_data.csv', encoding='cp932', low_memory=False)
    except Exception:
        df_m_auto = pd.read_csv('keiba_master_data.csv', encoding='utf-8-sig', low_memory=False)

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
    with col_p2:
        p_track = st.selectbox("トラック", ["芝", "ダート", "障害"], key="p_track")
        p_distance = st.number_input("距離 (m)", value=2000, step=100, key="p_distance")
    with col_p3:
        p_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="p_condition")

    st.markdown("---")
    st.markdown("### 📋 出馬表テキストの一発ペースト")
    st.info("💡 **使い方**: netkeibaの出馬表からテキストをコピーして下に貼り付けてな！馬番を自動認識するで。")
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
                jockey = "不明"
                weight = 56.0
               
                for j in range(i+1, min(i+6, len(cleaned_lines))):
                    sub_line = cleaned_lines[j]
                    if "データベース" in sub_line:
                        h_name = sub_line.replace("のデータベース", "").strip()
                    elif any(s in sub_line for s in ["牡", "牝", "セ"]) and len(sub_line) <= 5:
                        pass
                    elif "人気" in sub_line or "(" in sub_line:
                        pass
                    try:
                        val = float(sub_line)
                        if 0 < val < 500:
                            odds = val
                    except ValueError:
                        if len(sub_line) >= 2 and not any(c.isdigit() for c in sub_line) and "人気" not in sub_line:
                            if jockey == "不明":
                                jockey = sub_line

                current_horse = {
                    'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                    'waku': waku, 'umaban': umaban, 'name': h_name, 'sex': '牡', 'age': 4, 'sire': '不明',
                    'odds': odds, 'popularity': umaban, 'weight': weight, 'jockey': jockey,
                    'jockey_win_rate': jockey_win_rates.get(jockey, 0.08), 'blinker': 0
                }
            i += 1
       
        if current_horse:
            input_data_list.append(current_horse)

    if len(input_data_list) == 0:
        st.warning("⚠️ テキスト未入力のため、手動モードで8頭分のデフォルトを表示しています。")
        for i in range(8):
            input_data_list.append({
                'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                'waku': 1, 'umaban': i+1, 'name': f"馬番{i+1}", 'sex': '牡', 'age': 4, 'sire': '不明',
                'odds': 10.0, 'popularity': i+1, 'weight': 56.0, 'jockey': '不明', 'jockey_win_rate': 0.08, 'blinker': 0
            })
    else:
        st.success(f"✨ テキストから出走馬 **{len(input_data_list)}頭** を正確に検出しました！")

    if model is None:
        st.error("⚠️ AIモデルが見つかりません。「AI再学習」タブからモデルを作成してください。")
    elif st.button("🚀 ガチ予測を実行する！"):
        df_input = pd.DataFrame(input_data_list)
        df_full = pd.concat([df_m_auto, df_input], ignore_index=True) if not df_m_auto.empty else df_input
        df_full = df_full.loc[:, ~df_full.columns.duplicated()]

        for col in ['place', 'track', 'condition', 'sire']:
            if col in df_full.columns:
                le = LabelEncoder()
                df_full[col] = le.fit_transform(df_full[col].astype(str))

        df_input_encoded = df_full.tail(len(df_input)).copy()
        features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
        for feat in features:
            if feat not in df_input_encoded.columns:
                df_input_encoded[feat] = 0

        X_pred = df_input_encoded[features].fillna(0)
        try:
            st.balloons()
            st.subheader("🎯 ガチAI予測結果ランキング")
            probs = model.predict_proba(X_pred)[:, 1]
            df_input['win_prob'] = probs * 100
            df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
            for idx, row in df_input.iterrows():
                st.write(f"**第 {idx+1} 位**: 馬番 {row['umaban']} 🐴 **{row['name']}** (予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍 / 騎手: {row['jockey']})")
        except Exception as e:
            st.warning(f"⚠️ エラー: {e}")

with tab2:
    st.subheader("📝 レース結果をマスターに追加する")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
        race_num = st.number_input("レース番号", min_value=1, max_value=12, value=11, key="r_num")
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
            r_name = st.text_input("馬名", f"馬番{u_num}", key=f"r_name_{i}")
            r_rank = st.number_input("確定着順", min_value=1, max_value=18, value=1, key=f"r_rank_{i}")
            r_odds = st.number_input("単勝オッズ", value=10.0, min_value=0.0, key=f"r_odds_{i}")
            new_data_list.append({
                'place': race_place, 'track': track_type, 'distance': distance, 'condition': condition,
                'waku': ((u_num-1)//2)+1, 'umaban': u_num, 'name': r_name, 'sex': '牡', 'age': 4,
                'jockey': '不明', 'sire': '不明', 'weight': 56.0, 'rank': r_rank, 'odds': r_odds, 'popularity': u_num
            })

    if st.button("🚀 追加データをマスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        df_combined = pd.concat([df_m_auto, df_new], ignore_index=True) if not df_m_auto.empty else df_new
        df_combined.to_csv('keiba_master_data.csv', index=False, encoding='cp932')
        st.balloons()
        st.success("🎉 マスターデータに結果が保存されました！")

with tab3:
    st.subheader("🧠 ガチAIを再学習させる")
    if not df_m_auto.empty:
        st.success(f"📂 マスターデータ行数: **{len(df_m_auto):,} 行**")
        if st.button("🚀 フルデータでAIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
                df_train = df_m_auto.copy().loc[:, ~df_m_auto.columns.duplicated()]
                if len(df_train) > 50000: df_train = df_train.sample(n=50000, random_state=42)
               
                df_train['target'] = (pd.to_numeric(df_train['rank'], errors='coerce') == 1).astype(int)
                for col in ['place', 'track', 'condition', 'sire']:
                    if col in df_train.columns:
                        df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))
               
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'place', 'track', 'condition', 'sire', 'blinker']
                for f in features:
                    if f not in df_train.columns: df_train[f] = 0
               
                X = df_train[features].fillna(0)
                y = df_train['target']
                clf = lgb.LGBMClassifier(random_state=42)
                clf.fit(X, y)
                joblib.dump(clf, 'keiba_ai_model.pkl')
                st.balloons()
                st.success("🎉 再学習が完了しました！")
            except Exception as e:
                st.warning(f"⚠️ 学習エラー: {e}")
    else:
        st.warning("⚠️ マスターデータが見つかりません。")
