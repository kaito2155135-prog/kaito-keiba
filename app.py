import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.title("🐎 競馬AI 勝ち馬予測＆データ蓄積アプリ")
st.write("スマホから出馬データを入力して予測し、レース後は結果を記録してAIを育てよう！")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# タブで「予測モード」と「結果の記録・データ蓄積モード」を切り替えられるようにする
tab1, tab2 = st.tabs(["🚀 勝ち馬予測", "📝 レース結果の記録・蓄積"])

with tab1:
    if model is None:
        st.error("⚠️ AIモデル（keiba_ai_model.pkl）が見つかりません。")
    else:
        st.success("✨ AIモデル稼働中！")

        num_horses = st.slider("出馬頭数を選んでください", min_value=1, max_value=18, value=8, key="pred_num")
       
        input_data_list = []
        for i in range(num_horses):
            with st.expander(f"馬番 {i+1} の情報", expanded=(i < 3)):
                col1, col2 = st.columns(2)
                with col1:
                    horse_name = st.text_input(f"馬名 ({i+1}頭目)", f"馬名{i+1}", key=f"p_name_{i}")
                    odds = st.number_input(f"単勝オッズ ({i+1}頭目)", value=10.0, min_value=1.0, key=f"p_odds_{i}")
                with col2:
                    popularity = st.number_input(f"人気順 ({i+1}頭目)", value=i+1, min_value=1, step=1, key=f"p_pop_{i}")
                    weight = st.number_input(f"斤量 ({i+1}頭目)", value=56.0, key=f"p_weight_{i}")
               
                input_data_list.append({
                    'name': horse_name,
                    'odds': odds,
                    'popularity': popularity,
                    'weight': weight
                })

        if st.button("🚀 このメンバーで勝率を予測する！"):
            df_input = pd.DataFrame(input_data_list)
            X_pred = df_input[['odds', 'popularity', 'weight']]

            try:
                st.balloons()
                st.subheader("🎯 AI予測結果ランキング")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_pred)[:, 1]
                    df_input['win_prob'] = probs * 100
                    df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)

                    for idx, row in df_input.iterrows():
                        st.write(f"**第 {idx+1} 位**: 🐴 **{row['name']}** (予測勝率: **{row['win_prob']:.2f}%** / オッズ: {row['odds']}倍)")
                else:
                    preds = model.predict(X_pred)
                    df_input['pred'] = preds
                    for idx, row in df_input.iterrows():
                        st.write(f"🐴 **{row['name']}** - 予測結果: {row['pred']}")
            except Exception as e:
                st.warning("⚠️ エラーが発生しました:")
                st.write(e)

with tab2:
    st.subheader("📊 レース結果の入力とデータ蓄積")
    st.write("レースが終わったら、実際の着順・タイム・通過順を入力して、自分の蓄積用データ（CSV）として保存しよう！")

    race_name = st.text_input("レース名（例: 2026年〇〇賞）", "第〇レース")
    result_num_horses = st.slider("出走頭数", min_value=1, max_value=18, value=8, key="res_num")

    result_data_list = []
    for i in range(result_num_horses):
        with st.expander(f"【結果入力】馬番 {i+1}", expanded=False):
            r_name = st.text_input(f"馬名 ({i+1}頭目)", f"馬名{i+1}", key=f"r_name_{i}")
            r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=i+1, key=f"r_rank_{i}")
            r_time = st.text_input(f"走破タイム (例: 1:33.2)", "1:35.0", key=f"r_time_{i}")
            r_corner = st.text_input(f"通過順 (例: 3-3-2)", "1-1", key=f"r_corner_{i}")
            r_odds = st.number_input(f"単勝オッズ", value=10.0, key=f"r_odds_{i}")
            r_pop = st.number_input(f"人気", value=i+1, key=f"r_pop_{i}")
            r_weight = st.number_input(f"斤量", value=56.0, key=f"r_weight_{i}")

            result_data_list.append({
                'race_name': race_name,
                'horse_name': r_name,
                'rank': r_rank,
                'time': r_time,
                'corner': r_corner,
                'odds': r_odds,
                'popularity': r_pop,
                'weight': r_weight
            })

    if st.button("📥 蓄積用データ（CSV）としてダウンロードする"):
        df_results = pd.DataFrame(result_data_list)
        csv_data = df_results.to_csv(index=False).encode('utf-8-sig')
       
        st.success("✨ 蓄積データの作成に成功しました！下のボタンからダウンロードして保存してください。")
        st.download_button(
            label="💾 レース結果CSVをダウンロード",
            data=csv_data,
            file_name=f"{race_name}_result.csv",
            mime="text/csv",
        )
        st.info("💡 溜まったCSVデータを後でまとめてAIの再学習（アップデート）に使うことで、どんどん予測精度を上げていけるで！")
