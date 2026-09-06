import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="穴馬特化型 競馬AI予想アプリ", layout="wide")

st.title("🎯 穴馬特化型 競馬AI予測システム")
st.markdown("条件変更 × 展開不利 × 同レース好走馬多数 × 単勝20倍以上 の激走馬を狙い撃つ専用アプリです。")

# ---------------------------------------------------------
# 1. マスターデータのロードと前処理（穴馬用拡張）
# ---------------------------------------------------------
@st.cache_data
def load_master_data():
    # ※ここに実際のCSVやJRA-VANデータ読み込み処理が入ります
    # サンプルとしてダミーのデータフレームを作成して返します
    np.random.seed(42)
    n_samples = 1000
    df_m = pd.DataFrame({
        'race_id': [f"R_{i%100:03d}" for i in range(n_samples)],
        'name': [f"馬A_{i%50}" for i in range(n_samples)],
        'year': 2025,
        'month': np.random.randint(1, 13, n_samples),
        'day': np.random.randint(1, 29, n_samples),
        'rank': np.random.randint(1, 19, n_samples),
        'odds': np.random.uniform(1.5, 100.0, n_samples),
        'place': np.random.choice(['東京', '中山', '阪神', '京都'], n_samples),
        'track': np.random.choice(['芝1600', 'ダ1200', '芝2000', 'ダ1800'], n_samples),
        'true_reverse_gap': np.random.uniform(-5.0, 5.0, n_samples),
        'sex': np.random.choice(['牡', '牝', 'セン'], n_samples),
    })
    jockey_win_rates = {}
    return df_m, jockey_win_rates

def calculate_member_level_score(df_m):
    if 'race_id' not in df_m.columns or 'rank' not in df_m.columns or 'name' not in df_m.columns:
        return df_m
    
    df_sorted = df_m.sort_values(by=['name', 'year', 'month', 'day'])
    df_sorted['next_rank'] = df_sorted.groupby('name')['rank'].shift(-1)
    
    next_good_runs = df_sorted[df_sorted['next_rank'] <= 3].groupby('race_id').size().reset_index(name='strong_rivals_count')
    
    df_m = df_m.merge(next_good_runs, on='race_id', how='left')
    df_m['strong_rivals_count'] = df_m['strong_rivals_count'].fillna(0)
    return df_m

df_m_auto, jockey_win_rates = load_master_data()
df_m_auto = calculate_member_level_score(df_m_auto)

# ---------------------------------------------------------
# サイドバー：予測対象レースの設定
# ---------------------------------------------------------
st.sidebar.header("レース条件入力")
p_place = st.sidebar.selectbox("開催場", ['東京', '中山', '阪神', '京都', '中京', '新潟'])
p_track = st.sidebar.selectbox("トラック", ['芝1600', '芝2000', 'ダ1200', 'ダ1800'])

tab1, tab2, tab3 = st.tabs(["🚀 穴馬AI予想メイン", "📊 入力データ確認", "🧠 AIモデル再学習"])

with tab1:
    st.subheader("出走馬データの入力・インポート")
    uploaded_file = st.file_uploader("netkeibaやJRA-VANのテキスト・CSVデータをアップロード", type=['csv', 'txt'])
    
    # サンプル用の一覧入力フォームの代替として簡易データ作成
    if st.button("🚀 穴馬激走予測を実行する"):
        # プレースホルダーとしての入力データフレーム
        df_input = pd.DataFrame({
            'name': ['テイエムサクセス', 'マイネルダビデ', 'シゲルカゼノオ', 'スマートレイチェル'],
            'odds': [45.2, 18.5, 82.1, 4.2], # 4.2倍（本命）はペナルティ対象になるかテスト
            'prev_rank': [8.0, 12.0, 7.0, 2.0],
            'is_course_changed': [True, True, True, False],
            'true_reverse_gap': [-3.2, -1.8, -4.5, 0.5],
            'prev_strong_rivals': [3.0, 2.0, 4.0, 0.0]
        })
        
        # ダミーモデルとエンコーダーの用意
        features = ['prev_rank', 'is_course_changed', 'true_reverse_gap', 'prev_strong_rivals']
        X_dummy = df_input[features].fillna(0)
        y_dummy = np.random.choice([0, 1], len(df_input))
        
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_dummy, y_dummy)
        
        # --- 穴馬特化型 スコアリングロジック ---
        model_probs = model.predict_proba(X_dummy)[:, 1]
        
        odds_penalty = np.where(df_input['odds'] < 20.0, -100.0, 0.0)
        prev_rank_bonus = np.where(df_input['prev_rank'] >= 6.0, 3.0, 0.0)
        course_change_bonus = np.where(df_input['is_course_changed'], 5.0, 0.0)
        unlucky_bias_bonus = np.where(df_input['true_reverse_gap'] < 0, abs(df_input['true_reverse_gap']) * 2.0, 0.0)
        strong_rival_bonus = df_input['prev_strong_rivals'] * 4.0
        
        df_input['穴馬期待度スコア'] = (model_probs * 10.0) + prev_rank_bonus + course_change_bonus + unlucky_bias_bonus + strong_rival_bonus + odds_penalty
        
        df_sorted_res = df_input.sort_values(by='穴馬期待度スコア', ascending=False).reset_index(drop=True)
        
        st.success("予測完了！条件を満たす激走穴馬を抽出しました。")
        st.dataframe(df_sorted_res[['name', 'odds', 'prev_rank', 'is_course_changed', 'prev_strong_rivals', '穴馬期待度スコア']])

with tab2:
    st.subheader("マスターデータ状況")
    st.write(f"読込レコード数: {len(df_m_auto)}件")
    st.dataframe(df_m_auto.head(10))

with tab3:
    st.subheader("AI再学習（穴馬特化ラベル）の設定")
    st.markdown("正解ラベルを **「単勝20倍以上かつ3着以内」** に切り替えてモデルを再学習します。")
    
    if st.button("穴馬専用モデルの学習を実行"):
        df_train = df_m_auto.copy()
        is_top3 = pd.to_numeric(df_train['rank'], errors='coerce') <= 3
        is_longshot = pd.to_numeric(df_train['odds'], errors='coerce') >= 20.0
        df_train['target'] = (is_top3 & is_longshot).astype(int)
        
        st.info(f"学習データ件数: {len(df_train)}, 穴馬正解データ数: {df_train['target'].sum()}件")
        st.success("穴馬特化型モデルのチューニング準備が完了しました。")
