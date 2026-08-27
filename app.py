import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import io

st.title("🐎 スマホで育てる！競馬AIマスターアプリ")
st.write("レース条件もしっかり入力して、AIに正確な予測をさせよう！")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# タブで機能を分ける
tab1, tab2, tab3 = st.tabs(["🚀 勝ち馬予測", "📝 レース結果をマスターに追加", "🧠 AIの再学習（アップデート）"])

with tab1:
    st.subheader("🚀 勝ち馬の予測（レース条件 ＋ スマート入力）")
   
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
        st.error("⚠️ AIモデル（keiba_ai_model.pkl）が見つかりません。「AIの再学習」タブから最初のモデルを作るか、ファイルをアップロードしてください。")
    else:
        st.success("✨ AIモデル稼働中！")

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
                    weight = st.number_input(f"斤量", value=56.0, key=f"p_weight_{i}")
                with col2:
                    odds = st.number_input(f"単勝オッズ", value=10.0, min_value=1.0, key=f"p_odds_{i}")
                    popularity = st.number_input(f"人気順", value=auto_umaban, min_value=1, step=1, key=f"p_pop_{i}")
                    r_jockey = st.text_input(f"騎手名", "騎手名", key=f"p_jockey_{i}")
               
                input_data_list.append({
                    'distance': p_distance,
                    'waku': auto_waku,
                    'umaban': auto_umaban,
                    'name': horse_name,
                    'sex': r_sex,
                    'age': r_age,
                    'odds': odds,
                    'popularity': popularity,
                    'weight': weight,
                    'jockey': r_jockey
                })

        if st.button("🚀 この条件とメンバーで勝率を予測する！"):
            df_input = pd.DataFrame(input_data_list)
           
            # AIが学習時に使う数値特徴量（距離も追加！）
            X_pred = df_input[['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance']].fillna(0)

            try:
                st.balloons()
                st.subheader("🎯 AI予測結果ランキング")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_pred)[:, 1]
                    df_input['win_prob'] = probs * 100
                    df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

                    for idx, row in df_input.iterrows():
                        st.write(f"**第 {idx+1} 位**: 馬番 {row['umaban']} 🐴 **{row['name']}** (予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍 / 騎手: {row['jockey']})")
                else:
                    preds = model.predict(X_pred)
                    df_input['pred'] = preds
                    for idx, row in df_input.iterrows():
                        st.write(f"馬番 {row['umaban']} 🐴 **{row['name']}** - 予測結果: {row['pred']}")
            except Exception as e:
                st.warning("⚠️ エラーが発生しました（AIモデルの期待する列名と違う可能性があります）:")
                st.write(e)

with tab2:
    st.subheader("📊 1つのマスターCSVにレース結果を蓄積する")
    st.write("すでにある「マスターCSV」をアップロードし、今回のレース結果を追記して『上書き保存（ダウンロード）』しよう！")

    master_file = st.file_uploader("📂 今ある『マスターCSVファイル』をアップロード（初回は不要です）", type=["csv"], key="master_up")

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
                r_sire = st.text_input(f"父馬名", "父馬名", key=f"r_sire_{i}")
                r_dam = st.text_input(f"母馬名", "母馬名", key=f"r_dam_{i}")
                r_weight = st.number_input(f"斤量", value=56.0, key=f"r_weight_{i}")
                r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=auto_umaban, key=f"r_rank_{i}")
                r_time = st.text_input(f"走破タイム (例: 1:33.2)", "1:35.0", key=f"r_time_{i}")
                r_corner = st.text_input(f"通過順 (例: 3-3-2)", "1-1", key=f"r_corner_{i}")
                r_odds = st.number_input(f"単勝オッズ", value=10.0, key=f"r_odds_{i}")
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

    if st.button("📥 マスターCSVにデータを追加してダウンロード"):
        df_new = pd.DataFrame(new_data_list)
       
        if master_file is not None:
            df_master = pd.read_csv(master_file, encoding='utf-8-sig')
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new

        csv_data = df_combined.to_csv(index=False).encode('utf-8-sig')
       
        st.success(f"✨ 追加完了！全 {len(df_combined)} 行のデータになりました。下のボタンから保存してください。")
        st.download_button(
            label="💾 最新の『keiba_master_data.csv』をダウンロード",
            data=csv_data,
            file_name="keiba_master_data.csv",
            mime="text/csv",
        )
        st.info("💡 スマホの同じファイル名（keiba_master_data.csv）で上書き保存していくことで、常に1つのファイルにデータが蓄積されていくで！")

with tab3:
    st.subheader("🧠 蓄積したマスターCSVでAIを再学習させる")
    st.write("これまで育ててきた『keiba_master_data.csv』をアップロードして、AI（LightGBM）をその場で賢くアップデートしよう！")

    train_file = st.file_uploader("📂 学習用『keiba_master_data.csv』をアップロード", type=["csv"], key="train_up")

    if train_file is not None:
        df_train = pd.read_csv(train_file, encoding='utf-8-sig')
        st.write(f"📊 読み込んだデータ数: **{len(df_train)} 行**")

        if st.button("🚀 このデータでAIを再学習・更新する！"):
            try:
                import lightgbm as lgb
               
                df_train['target'] = (df_train['rank'] == 1).astype(int)
                # 再学習時にも距離を特徴量に含める
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance']
               
                X = df_train[features].fillna(0)
                y = df_train['target']

                if len(X) < 5:
                    st.warning("⚠️ データ数が少なすぎます（最低5行以上必要です）。もっとレース結果を蓄積してね！")
                else:
                    clf = lgb.LGBMClassifier(random_state=42)
                    clf.fit(X, y)

                    model_path = 'keiba_ai_model.pkl'
                    joblib.dump(clf, model_path)

                    st.balloons()
                    st.success("🎉 AIの再学習が完了しました！新しいAIモデルが誕生しました！")
                   
                    with open(model_path, "rb") as f:
                        st.download_button(
                            label="💾 新しいAIモデル（keiba_ai_model.pkl）をダウンロード",
                            data=f,
                            file_name="keiba_ai_model.pkl",
                            mime="application/octet-stream"
                        )
                    st.info("💡 ダウンロードした『keiba_ai_model.pkl』をGitHubにアップロードし直せば、アプリのAIがさらに賢くなったものにアップデートされるで！")

            except Exception as e:
                st.error("⚠️ 学習中にエラーが発生しました:")
                st.write(e)
