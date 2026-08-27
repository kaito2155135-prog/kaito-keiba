import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.title("🐎 競馬AI 勝ち馬予測＆データ蓄積アプリ")
st.write("ターゲット仕様の細かなデータを蓄積して、自分だけのAIを育てよう！")

@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

tab1, tab2 = st.tabs(["🚀 勝ち馬予測", "📝 レース結果の記録・蓄積"])

with tab1:
    if model is None:
        st.error("⚠️ AIモデル（keiba_ai_model.pkl）が見つかりません。")
    else:
        st.success("✨ AIモデル稼働中！")

        num_horses = st.slider("予測する出馬頭数", min_value=1, max_value=18, value=8, key="pred_num")
       
        input_data_list = []
        for i in range(num_horses):
            with st.expander(f"馬番 {i+1} の予測用データ", expanded=(i < 3)):
                col1, col2 = st.columns(2)
                with col1:
                    horse_name = st.text_input(f"馬名 ({i+1}頭目)", f"馬名{i+1}", key=f"p_name_{i}")
                    odds = st.number_input(f"単勝オッズ", value=10.0, min_value=1.0, key=f"p_odds_{i}")
                with col2:
                    popularity = st.number_input(f"人気順", value=i+1, min_value=1, step=1, key=f"p_pop_{i}")
                    weight = st.number_input(f"斤量", value=56.0, key=f"p_weight_{i}")
               
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
    st.subheader("📊 ターゲット仕様：レース結果と詳細データの蓄積")
    st.write("レースの全体条件と、各馬の詳細データを入力してCSVとして保存しよう！")

    # --- レース全体の基本情報 ---
    st.markdown("### 🏟️ レース条件の選択")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        race_date = st.date_input("開催日", datetime.date(2026, 6, 1))
        race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"])
    with col_r2:
        race_name = st.text_input("レース名", "〇〇ステークス")
        track_type = st.selectbox("トラック", ["芝", "ダート", "障害"])
    with col_r3:
        distance = st.number_input("距離 (m)", value=1600, step=100)
        condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])
        weather = st.selectbox("天気", ["晴", "曇", "雨", "小雨", "雪"])

    st.markdown("---")
    res_num_horses = st.slider("出走頭数", min_value=1, max_value=18, value=8, key="res_num")

    result_data_list = []
    for i in range(res_num_horses):
        with st.expander(f"【詳細データ入力】馬番 {i+1}", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                r_waku = st.number_input(f"枠番", min_value=1, max_value=8, value=1, key=f"r_waku_{i}")
                r_umaban = st.number_input(f"馬番", min_value=1, max_value=18, value=i+1, key=f"r_umaban_{i}")
                r_name = st.text_input(f"馬名", f"馬名{i+1}", key=f"r_name_{i}")
                r_jockey = st.text_input(f"騎手名", "騎手名", key=f"r_jockey_{i}")
                r_weight = st.number_input(f"斤量", value=56.0, key=f"r_weight_{i}")
            with col_b:
                r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=i+1, key=f"r_rank_{i}")
                r_time = st.text_input(f"走破タイム (例: 1:33.2)", "1:35.0", key=f"r_time_{i}")
                r_corner = st.text_input(f"通過順 (例: 3-3-2)", "1-1", key=f"r_corner_{i}")
                r_odds = st.number_input(f"単勝オッズ", value=10.0, key=f"r_odds_{i}")
                r_pop = st.number_input(f"人気順", value=i+1, key=f"r_pop_{i}")

            # データを辞書にまとめる（ターゲットの項目を意識した形）
            result_data_list.append({
                'date': race_date,
                'place': race_place,
                'race_name': race_name,
                'track': track_type,
                'distance': distance,
                'condition': condition,
                'weather': weather,
                'waku': r_waku,
                'umaban': r_umaban,
                'horse_name': r_name,
                'jockey': r_jockey,
                'weight': r_weight,
                'rank': r_rank,
                'time': r_time,
                'corner': r_corner,
                'odds': r_odds,
                'popularity': r_pop
            })

    if st.button("📥 ターゲット仕様の蓄積データ（CSV）をダウンロード"):
        df_results = pd.DataFrame(result_data_list)
        csv_data = df_results.to_csv(index=False).encode('utf-8-sig')
       
        file_title = f"{race_date}_{race_place}_{race_name}"
        st.success("✨ 蓄積データの作成に成功しました！下のボタンから保存してください。")
        st.download_button(
            label="💾 詳細レース結果CSVをダウンロード",
            data=csv_data,
            file_name=f"{file_title}_result.csv",
            mime="text/csv",
        )
        st.info("💡 これで日付、場所、馬場、枠番、騎手などのターゲット同等のデータが丸ごとCSVに保存できるようになりました！")
