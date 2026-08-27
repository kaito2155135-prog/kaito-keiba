import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

st.set_page_config(page_title="競馬AI予測アプリ", page_icon="🐎", layout="centered")

st.title("🐎 競馬AI予測アプリ（一括コピペ対応版）")
st.write("netkeibaなどの出馬表をコピーして貼り付けるだけで、一瞬で全頭の予測ができます！")

# モデルとマスターデータの読み込み
@st.cache_resource
def load_model_and_data():
    model = None
    if os.path.exists("keiba_ai_model.pkl"):
        model = joblib.load("keiba_ai_model.pkl")
   
    master_df = pd.DataFrame()
    part1_exists = os.path.exists("keiba_master_data_part1.csv")
    part2_exists = os.path.exists("keiba_master_data_part2.csv")
   
    if part1_exists and part2_exists:
        df1 = pd.read_csv("keiba_master_data_part1.csv")
        df2 = pd.read_csv("keiba_master_data_part2.csv")
        master_df = pd.concat([df1, df2], ignore_index=True)
    elif part1_exists:
        master_df = pd.read_csv("keiba_master_data_part1.csv")
    elif part2_exists:
        master_df = pd.read_csv("keiba_master_data_part2.csv")
       
    return model, master_df

model, master_df = load_model_and_data()

if model is None:
    st.error("⚠️ 予測モデル（keiba_ai_model.pkl）が見つかりません。GitHubにアップロードされているか確認してください。")
else:
    st.success("✅ AIモデルの読み込み完了！")

    tab1, tab2 = st.tabs(["📝 テキスト一括貼り付け予測", "⚙️ 従来の個別入力"])

    with tab1:
        st.subheader("出馬表テキストの一括貼り付け")
        st.info("💡 ネット競馬などの出馬表ページで、出走馬の部分をマウスでガーッと選択してコピーし、下のボックスにそのまま貼り付けてください。")
       
        raw_text = st.text_area("ここにコピーした出馬表を貼り付け", height=200, placeholder="例: 1 1 　コントレイル 牡3 56.0 福永祐一 1.8 ...")
       
        # 簡易的なオッズや馬名抽出の処理
        if st.button("🚀 貼り付けたデータで一括予測する"):
            if not raw_text.strip():
                st.warning("⚠️ テキストが入力されていません。")
            else:
                # テキストから行ごとに分割
                lines = raw_text.split("\n")
                parsed_data = []
               
                for line in lines:
                    # 数字や文字の羅列から馬名や数字っぽいものを探す簡易パーサー
                    # （完全ではないですが、オッズや馬番のパターンを正規表現や数値抽出で拾います）
                    numbers = re.findall(r'\d+\.\d+|\d+', line)
                    # 馬名っぽい文字列（ひらがな、カタカナ、漢字、アルファベット）を探す
                    words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FA5A-Za-z]+', line)
                   
                    if words:
                        # 抽出できたデータから仮の数値を割り当てて予測用データを作る
                        horse_name = words[0] if len(words) > 0 else "不明馬"
                        # オッズらしき数値（小数点を含むもの）があれば取得、なければデフォルト30.0
                        odds = float(numbers[-1]) if numbers and '.' in numbers[-1] else 30.0
                        wakuban = int(numbers[0]) if numbers else 1
                        umaban = int(numbers[1]) if len(numbers) > 1 else 1
                       
                        parsed_data.append({
                            "馬名": horse_name,
                            "枠番": wakuban,
                            "馬番": umaban,
                            "オッズ": odds
                        })
               
                if parsed_data:
                    df_input = pd.DataFrame(parsed_data)
                    st.write("🔍 **読み取った出走馬リスト:**", df_input)
                   
                    # 特徴量の調整（モデルの必要とする形に合わせる）
                    # ここでは簡易的にマスターデータの平均値などを使って予測用行列を作る
                    X_pred = pd.DataFrame(index=df_input.index)
                    X_pred['odds'] = df_input['オッズ']
                    X_pred['wakuban'] = df_input['枠番']
                    X_pred['umaban'] = df_input['馬番']
                   
                    # モデルが要求するカラム数に合わせるため、足りない分を0や平均で補う
                    if hasattr(model, "feature_names_in_"):
                        for col in model.feature_names_in_:
                            if col not in X_pred.columns:
                                X_pred[col] = 0.0
                        X_pred = X_pred[model.feature_names_in_]

                    # 予測実行
                    try:
                        preds = model.predict_proba(X_pred)[:, 1]
                        df_input['AI予測スコア'] = preds
                        df_input = df_input.sort_values(by="AI予測スコア", ascending=False).reset_index(drop=True)
                        df_input['順位'] = df_input.index + 1
                       
                        st.balloons()
                        st.subheader("🏆 予測結果ランキング")
                        st.dataframe(df_input[['順位', '馬名', '馬番', 'オッズ', 'AI予測スコア']])
                       
                    except Exception as e:
                        st.error(f"予測計算中にエラーが発生しました: {e}")
                else:
                    st.error("⚠️ テキストから有効な馬情報を読み取れませんでした。別の形式でコピーするか、個別入力タブをお試しください。")

    with tab2:
        st.subheader("従来の個別入力モード")
        st.write("従来どおり手動で数値を調整したい場合はこちらをお使いください。")
        # 従来の簡易入力
        user_odds = st.number_input("単勝オッズ", min_value=1.0, max_value=999.0, value=5.0)
        user_umaban = st.number_input("馬番", min_value=1, max_value=18, value=1)
       
        if st.button("個別予測する"):
            # ダミーの1頭分予測
            sample_df = pd.DataFrame({'odds': [user_odds], 'umaban': [user_umaban]})
            if hasattr(model, "feature_names_in_"):
                for col in model.feature_names_in_:
                    if col not in sample_df.columns:
                        sample_df[col] = 0.0
                sample_df = sample_df[model.feature_names_in_]
           
            try:
                score = model.predict_proba(sample_df)[0][1]
                st.success(f"この馬のAI予測スコア: {score:.4f}")
            except Exception as e:
                st.error(f"エラー: {e}")