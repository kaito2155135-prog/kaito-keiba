import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.preprocessing import LabelEncoder

st.title("🐎【ガチ仕様】スマホで育てる！競馬AIマスターアプリ")
st.write("絶対安心・エラー回避ガード搭載版！過去5年分データ合体＆ブリンカー対応や！🔥")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# --- データの安全読み込み ＆ もし無ければ自動生成する安全ガード ---
MASTER_FILENAME = 'keiba_master_data.csv'
dfs = []

# ディレクトリ内からCSVを探す
for f in os.listdir('.'):
    if 'keiba_master_data' in f and f.endswith('.csv'):
        try:
            temp_df = pd.read_csv(f, encoding='utf-8-sig')
            if len(temp_df) > 0:
                dfs.append(temp_df)
        except Exception:
            pass

# 万が一ファイルが見つからない場合の自動ダミーデータ作成（アプリを止まらせないため！）
if len(dfs) == 0:
    # 応急処置用のダミーデータを作成
    dummy_data = {
        'date': ['2026-01-01']*10,
        'place': ['東京']*10,
        'race_number': [11]*10,
        'race_name': ['テストレース']*10,
        'track': ['芝']*10,
        'distance': [1600]*10,
        'condition': ['良']*10,
        'weather': ['晴']*10,
        'waku': [1,2,3,4,5,6,7,8,1,2],
        'umaban': [1,2,3,4,5,6,7,8,9,10],
        'horse_name': [f'テスト馬{i}' for i in range(1, 11)],
        'sex': ['牡']*10,
        'age': [3]*10,
        'jockey': ['武豊']*10,
        'trainer': ['調教師A']*10,
        'stable': ['栗東']*10,
        'sire': ['ディープインパクト']*10,
        'dam': ['母馬A']*10,
        'weight': [56.0]*10,
        'blinker': ['', 'B', '', '', '', 'B', '', '', '', ''],
        'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'time': ['1:35.0']*10,
        'corner': ['1-1']*10,
        'odds': [2.5, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
        'popularity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    df_m_auto = pd.DataFrame(dummy_data)
    df_m_auto.to_csv(MASTER_FILENAME, index=False, encoding='utf-8-sig')
else:
    df_m_auto = pd.concat(dfs, ignore_index=True)

if 'blinker' not in df_m_auto.columns:
    df_m_auto['blinker'] = ""

jockey_win_rates = {}
if 'jockey' in df_m_auto.columns and 'rank' in df_m_auto.columns:
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
    st.success(f"📂 マスターデータ稼働中！（総データ数: {len(df_m_auto)}行✨）")

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
        st.warning("⚠️ AIモデルがまだありません。「ガチAIの再学習」タブから最初のモデルを作成してください！")
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
        if model is None:
            st.error("⚠️ まず「ガチAIの再学習」タブからモデルを作成してください！")
        else:
            df_input = pd.DataFrame(input_data_list)
            df_full = pd.concat([df_m_auto, df_input], ignore_index=True)

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
                st.warning("⚠️ エラーが発生しました（下の再学習を行ってください）:")
                st.write(e)

with tab2:
    st.subheader("📝 レース結果をマスターに追加する")
    st.write("ブリンカー情報も含めて結果を追加・蓄積するで！✨")

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

    if st.button("🚀 追加データをマスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        if os.path.exists(MASTER_FILENAME):
            df_master = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig')
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(MASTER_FILENAME, index=False, encoding='utf-8-sig')
        st.balloons()
        st.success(f"🎉 追加成功！マスターデータに結果が保存されました！✨")

with tab3:
    st.subheader("🧠 全データでガチAIを再学習させる")
    st.success(f"📂 学習データ準備OK！（総データ数: **{len(df_m_auto)} 行**）")

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

            target_train_df['target'] = (target_train_df['rank'] == 1).astype(int)
            features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
           
            X = target_train_df[features].fillna(0)
            y = target_train_df['target']

            clf = lgb.LGBMClassifier(random_state=42)
            clf.fit(X, y)

            joblib.dump(clf, 'keiba_ai_model.pkl')
            st.balloons()
            st.success("🎉 最強ガチAIの再学習が完了しました！これで予測もバッチリ動くで！画面をリロードして予測してみてな✨")

        except Exception as e:
            st.error("⚠️ 学習中にエラーが発生しました:")
            st.write(e)
