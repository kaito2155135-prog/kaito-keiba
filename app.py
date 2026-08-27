import streamlit as st
import pandas as pd
import joblib

# ページの設定
st.set_page_config(page_title="競馬AI予想アプリ", page_icon="🏇", layout="wide")

st.title("🏇 競馬AI 勝ち馬予測アプリ")
st.write("今週の出馬表ファイルをアップロードして、条件を選ぶだけでAIが勝率を予測します！")

# 1. 保存したAIモデル（頭脳）を読み込む
@st.cache_resource
def load_model():
    # 'keiba_ai_model.pkl' をアプリと同じフォルダに入れておく
    return joblib.load('keiba_ai_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ AIモデル（keiba_ai_model.pkl）が見つかりません。同じフォルダにアップロードしてください。エラー: {e}")
    st.stop()

# 2. タイムを秒数に変換する関数
def convert_time_to_seconds(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    parts = val_str.split('.')
    try:
        if len(parts) == 3:
            return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / (10 ** len(parts[2]))
        else:
            return float(val_str)
    except:
        return None

# 3. ファイルアップロード（CSV or Parquet）
uploaded_file = st.file_uploader("📂 今週の出馬表ファイル（CSVまたはParquet）をアップロード", type=['csv', 'parquet'])

if uploaded_file is not None:
    # 読み込み
    try:
        if uploaded_file.name.endswith('.parquet'):
            df_target = pd.read_parquet(uploaded_file)
        else:
            df_target = pd.read_csv(uploaded_file, encoding='shift_jis', low_memory=False)
    except Exception as e:
        df_target = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)

    st.success("🎉 ファイルの読み込みに成功しました！")

    # サイドバー（条件選択）
    st.sidebar.header("🔍 予想条件の絞り込み")
   
    # 馬場状態の選択肢（データにある場合）
    baba_options = ['指定なし']
    if '馬場状態' in df_target.columns:
        baba_options += list(df_target['馬場状態'].dropna().unique())
    selected_baba = st.sidebar.selectbox("馬場状態を選択", baba_options)

    # レース番号の選択肢
    race_options = [0]
    if 'レース番号' in df_target.columns:
        race_options = [0] + sorted(list(df_target['レース番号'].dropna().unique()))
    selected_race = st.sidebar.selectbox("レース番号を選択 (0=全レース)", race_options)

    # 特徴量の前処理
    features = [
        '距離', '頭数', '馬番', '斤量', '人気', '単勝オッズ',
        '走破タイム_秒', '上がり3Fタイム', '上がり3F順位', '馬体重', '馬体重増減'
    ]

    if '走破タイム_秒' not in df_target.columns and '走破タイム' in df_target.columns:
        df_target['走破タイム_秒'] = df_target['走破タイム'].apply(convert_time_to_seconds)
    elif '走破タイム_秒' not in df_target.columns:
        df_target['走破タイム_秒'] = 0.0

    for col in features:
        if col in df_target.columns:
            df_target[col] = pd.to_numeric(df_target[col], errors='coerce')
        else:
            df_target[col] = 0.0

    # 予測実行
    X_pred = df_target[features]
    df_target['pred_score'] = model.predict_proba(X_pred)[:, 1] * 100

    # 絞り込み適用
    if selected_baba != '指定なし' and '馬場状態' in df_target.columns:
        df_target = df_target[df_target['馬場状態'] == selected_baba]

    if selected_race > 0 and 'レース番号' in df_target.columns:
        df_target = df_target[df_target['レース番号'] == selected_race]

    # ランキング表示
    sorted_df = df_target.sort_values('pred_score', ascending=False)
   
    st.subheader(f"🏆 AI予想ランキング（条件：馬場={selected_baba}, レース={selected_race if selected_race>0 else '全レース'}）")
   
    display_cols = [c for c in ['レース番号', '馬番', '馬名', '馬場状態', '単勝オッズ', 'pred_score'] if c in sorted_df.columns]
   
    # 画面に綺麗にテーブル表示
    st.dataframe(sorted_df[display_cols].head(20), use_container_width=True)
else:
    st.info("👆 まずは上のボタンから、今週の出馬表ファイルをアップロードしてください！")
