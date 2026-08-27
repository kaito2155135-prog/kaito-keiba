import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
from sklearn.preprocessing import LabelEncoder

st.title("🐎【ガチ仕様】スマホで育てる！競馬AIマスターアプリ")
st.write("血統（父馬）や開催コース条件を完全網羅！本気で勝ちに行くためのガチAIモデルや！🔥")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# --- 共通：マスターデータの自動読み込み＆騎手勝率の計算 ---
MASTER_FILENAME = 'keiba_master_data.csv'
jockey_win_rates = {}

if os.path.exists(MASTER_FILENAME):
    try:
        df_m_auto = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig')
        if 'jockey' in df_m_auto.columns and 'rank' in df_m_auto.columns:
            jockey_stats = df_m_auto.groupby('jockey').agg(
                total=('rank', 'count'),
                wins=('rank', lambda x: (x == 1).sum())
            )
            for jock, row in jockey_stats.iterrows():
                if row['total'] > 0:
                    jockey_win_rates[jock] = row['wins'] / row['total']
    except Exception as e:
        pass

# タブで機能を分ける
tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果をマスターに追加", "🧠 ガチAIの再学習（アップデート）"])

with tab1:
    st.subheader("🚀 勝ち馬のガチ予測（コース・血統連動）")
   
    if os.path.exists(MASTER_FILENAME):
        st.success(f"📂 クラウド上のマスターデータ読み込み中（蓄積データ: {len(df_m_auto)}行 / 騎手データ反映済み✨）")
    else:
        st.info("💡 ヒント: まだマスターデータがありません。「レース結果をマスターに追加」から最初のデータを保存してね！")

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
        st.error("⚠️ AIモデル（keiba_ai_model.pkl）が見つかりません。「ガチAIの再学習」タブから最初のモデルを作るか、データを蓄積してください。")
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
               
                jock_rate = jockey_win_rates.get(r_jockey, 0.08)

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
                    'jockey_win_rate': jock_rate
                })

        if st.button("🚀 ガチ予測を実行する！"):
            df_input = pd.DataFrame(input_data_list)
           
            # 過去のマスターデータと結合して文字エンコーディングを一致させる
            if os.path.exists(MASTER_FILENAME):
                df_master_all = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig')
                df_full = pd.concat([df_master_all, df_input], ignore_index=True)
            else:
                df_full = df_input

            # カテゴリカル変数の数値化
            cat_cols = ['place', 'track', 'condition', 'sire']
            for col in cat_cols:
                le = LabelEncoder()
                df_full[col] = df_full[col].astype(str)
                df_full[col] = le.fit_transform(df_full[col])

            # 入力部分だけを切り出し
            df_input_encoded = df_full.tail(len(df_input)).copy()

            features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire']
            X_pred = df_input_encoded[features].fillna(0)

            try:
                st.balloons()
                st.subheader("🎯 ガチAI予測結果ランキング")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_pred)[:, 1]
                    df_input['win_prob'] = probs * 100
                    df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

                    for idx, row in df_input.iterrows():
                        st.write(f"**第 {idx+1} 位**: 馬番 {row['umaban']} 🐴 **{row['name']}** (父:{row['sire']} / 予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍 / 騎手: {row['jockey']} [勝率:{row['jockey_win_rate']*100:.1f}%])")
                else:
                    preds = model.predict(X_pred)
                    df_input['pred'] = preds
                    for idx, row in df_input.iterrows():
                        st.write(f"馬番 {row['umaban']} 🐴 **{row['name']}** (父:{row['sire']}) - 予測結果: {row['pred']}")
            except Exception as e:
                st.warning("⚠️ エラーが発生しました（学習時の特徴量と一致しない可能性があります。「ガチAIの再学習」を行ってください）:")
                st.write(e)

with tab2:
    st.subheader("📝 レース結果をマスターに直接追加する")
    st.write("血統やコース条件も含めて結果を保存し、AIをさらに賢くするで！✨")

    if os.path.exists(MASTER_FILENAME):
        df_current_check = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig')
        st.info(f"📊 現在のクラウド上の蓄積データ: **{len(df_current_check)} 行**")
    else:
        st.info("📊 現在の蓄積データ: **0 行（これが最初のデータになります）**")

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
                r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=auto_umaban, key=f"r_rank_{i}")
                r_time = st.text_input(f"走破タイム (例: 1:33.2)", "1:35.0", key=f"r_time_{i}")
                r_corner = st.text_input(f"通過順 (例: 3-3-2)", "1-1", key=f"r_corner_{i}")
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
                'rank': r_rank,
                'time': r_time,
                'corner': r_corner,
                'odds': r_odds,
                'popularity': r_pop
            })

    if st.button("🚀 クラウドのマスターCSVにこの結果を追加する！"):
        df_new = pd.DataFrame(new_data_list)
       
        if os.path.exists(MASTER_FILENAME):
            df_master = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig')
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(MASTER_FILENAME, index=False, encoding='utf-8-sig')
       
        st.balloons()
        st.success(f"🎉 追加成功！クラウド上のマスターデータが全 {len(df_combined)} 行に更新されました！✨")
        st.info("💡 ページを再読み込み（リロード）すると、最新データが反映されるで！")

with tab3:
    st.subheader("🧠 クラウド上のデータでガチAIを再学習させる")
   
    if os.path.exists(MASTER_FILENAME):
        target_train_df = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig')
        st.success(f"📂 クラウド上のマスターデータ (`{MASTER_FILENAME}`) を検出しました！")
        st.write(f"📊 学習に使えるデータ数: **{len(target_train_df)} 行**")

        if st.button("🚀 ガチAIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
               
                # 騎手勝率の計算
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

                # カテゴリカル変数を数値に変換
                cat_cols = ['place', 'track', 'condition', 'sire']
                for col in cat_cols:
                    if col in target_train_df.columns:
                        le = LabelEncoder()
                        target_train_df[col] = target_train_df[col].astype(str)
                        target_train_df[col] = le.fit_transform(target_train_df[col])
                    else:
                        target_train_df[col] = 0

                target_train_df['target'] = (target_train_df['rank'] == 1).astype(int)
               
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire']
               
                X = target_train_df[features].fillna(0)
                y = target_train_df['target']

                if len(X) < 5:
                    st.warning("⚠️ データ数が少なすぎます（最低5行以上必要です）。もっとレース結果を蓄積してね！")
                else:
                    clf = lgb.LGBMClassifier(random_state=42)
                    clf.fit(X, y)

                    model_path = 'keiba_ai_model.pkl'
                    joblib.dump(clf, model_path)

                    st.balloons()
                    st.success("🎉 ガチAIの再学習が完了しました！血統・コース条件を網羅した最強AIモデルが誕生！✨")

            except Exception as e:
                st.error("⚠️ 学習中にエラーが発生しました:")
                st.write(e)
    else:
        st.warning("⚠️ クラウド上にマスターデータがありません。「レース結果をマスターに追加」から最初のデータを保存してね！")