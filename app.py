import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import re
import urllib.request
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder

st.title("🐎【完全版】スマホで育てる！競馬AIマスターアプリ")
st.write("netkeibaのURLを貼り付けるだけで、全頭のデータを自動一括取得する最新版！✨🔥")

@st.cache_resource
def load_model():
    try:
        return joblib.load('keiba_ai_model.pkl')
    except Exception as e:
        return None

model = load_model()

# --- マスターデータの読み込みとカラム自動マッピング ---
jockey_win_rates = {}
df_m_auto = pd.DataFrame()
dfs = []

files = ['keiba_master_data_part1.csv', 'keiba_master_data_part2.csv', 'keiba_master_data.csv']

for f in files:
    if os.path.exists(f):
        try:
            try:
                temp_df = pd.read_csv(f, encoding='cp932', low_memory=False)
            except Exception:
                temp_df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
               
            if len(temp_df) > 10:
                dfs.append(temp_df)
        except Exception as e:
            pass

if len(dfs) > 0:
    df_m_auto = pd.concat(dfs, ignore_index=True)
   
    rename_cols = {}
    existing_cols = set(df_m_auto.columns)
   
    for col in df_m_auto.columns:
        c_str = str(col).strip()
        target_name = None
       
        if any(w in c_str for w in ['着順', '確定着順']):
            target_name = 'rank'
        elif any(w in c_str for w in ['騎手', '騎手名']):
            target_name = 'jockey'
        elif any(w in c_str for w in ['父', '血統(父)', '種牡馬']):
            target_name = 'sire'
        elif any(w in c_str for w in ['馬名']):
            target_name = 'name'
        elif any(w in c_str for w in ['オッズ', '単勝']):
            target_name = 'odds'
        elif any(w in c_str for w in ['人気', '順位']):
            target_name = 'popularity'
        elif any(w in c_str for w in ['斤量', '負担重量']):
            target_name = 'weight'
        elif any(w in c_str for w in ['年齢', '馬齢']):
            target_name = 'age'
        elif any(w in c_str for w in ['枠番', '枠']):
            target_name = 'waku'
        elif any(w in c_str for w in ['馬番']):
            target_name = 'umaban'
        elif any(w in c_str for w in ['距離']):
            target_name = 'distance'
        elif any(w in c_str for w in ['開催', '場所', '競馬場']):
            target_name = 'place'
        elif any(w in c_str for w in ['芝・ダート', 'トラック', 'コース']):
            target_name = 'track'
        elif any(w in c_str for w in ['馬場', '馬場状態']):
            target_name = 'condition'
           
        if target_name and target_name not in existing_cols and target_name not in rename_cols.values():
            if col != target_name:
                rename_cols[col] = target_name
   
    if rename_cols:
        df_m_auto = df_m_auto.rename(columns=rename_cols)

    df_m_auto = df_m_auto.loc[:, ~df_m_auto.columns.duplicated()]

    if 'blinker' not in df_m_auto.columns:
        df_m_auto['blinker'] = ""
   
    if 'jockey' in df_m_auto.columns and 'rank' in df_m_auto.columns:
        df_m_auto['rank'] = pd.to_numeric(df_m_auto['rank'], errors='coerce')
        jockey_stats = df_m_auto.groupby('jockey').agg(
            total=('rank', 'count'),
            wins=('rank', lambda x: (x == 1).sum())
        )
        for jock, row in jockey_stats.iterrows():
            if row['total'] > 0:
                jockey_win_rates[jock] = row['wins'] / row['total']

tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果をマスターに追加", "🧠 ガチAIの再学習"])

with tab1:
    st.subheader("🚀 勝ち馬のガチ予測（netkeiba URL自動取得版）")
   
    if len(df_m_auto) > 10:
        st.success(f"📂 マスターデータ読み込み成功！（総データ数: {len(df_m_auto):,}行✨）")
    else:
        st.warning(f"⚠️ データ数が少ない状態です（現在 {len(df_m_auto)}行）。")

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

    with st.expander("🌐 netkeibaの出馬表URLから自動取得", expanded=True):
        st.markdown("netkeibaの対象レース出馬表URLをここに貼り付けてな！")
       
        # サンプルとして直近のnetkeibaレースURLの形をプレースホルダーや初期値に提示
        netkeiba_url_input = st.text_input(
            "netkeiba 出馬表URL",
            value="https://race.netkeiba.com/race/shutuba.html?race_id=202605010111",
            placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=..."
        )
       
        if st.button("✨ URLから全頭データを自動取得する！"):
            if netkeiba_url_input.strip():
                try:
                    req = urllib.request.Request(
                        netkeiba_url_input,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req) as response:
                        html = response.read().decode('euc-jp', errors='ignore')
                   
                    soup = BeautifulSoup(html, 'html.parser')
                    parsed_horses = []
                   
                    # netkeibaの出馬表テーブル行を走査
                    tr_list = soup.select('tr.HorseList') or soup.find_all('tr', class_=re.compile('HorseList'))
                   
                    if not tr_list:
                        # 別の構造に対応
                        tr_list = soup.select('.ShutubaTable tr')

                    for tr in tr_list:
                        h_data = {}
                       
                        # 馬番
                        umaban_el = tr.select_T('.Umaban, .Num') if hasattr(tr, 'select_T') else tr.select('.Umaban')
                        if not umaban_el:
                            umaban_el = tr.find(class_=re.compile('Umaban|num'))
                       
                        # テキストベースで手堅く探す
                        text_all = tr.get_text()
                       
                        # 馬番の抽出 (1〜18)
                        m_umaban = re.search(r'\b([1-9]|1[0-8])\b', tr.get_text())
                       
                        # 馬名の抽出（tr内の<a>タグで特定のクラスや、カタカナを探す）
                        a_tags = tr.find_all('a', href=re.compile('horse'))
                        horse_name = ""
                        for a in a_tags:
                            candidate = a.get_text().strip()
                            if re.search(r'^[ァ-ンー]+$', candidate):
                                horse_name = candidate
                                break
                       
                        if not horse_name:
                            # ざっくりカタカナを探す
                            m_name = re.search(r'([ァ-ンー]{2,})', text_all)
                            if m_name:
                                horse_name = m_name.group(1)

                        # 枠番・馬番・オッズ・人気などの簡易パース
                        m_pop = re.search(r'([1-9][0-9]*)人気', text_all)
                        m_odds = re.search(r'([0-9]+\.[0-9])', text_all)
                        m_sex_age = re.search(r'([牡牝騸セ])([2-9])', text_all)
                        m_wt = re.search(r'(5[0-9]\.[0-9])', text_all)
                       
                        # 簡易的に行から各要素を拾う
                        if horse_name:
                            h_data['name'] = horse_name
                           
                            # 馬番推定（行の最初の方にある数字）
                            nums = re.findall(r'\b([1-9]|1[0-8])\b', text_all)
                            if nums:
                                u_num = int(nums[0])
                                h_data['umaban'] = u_num
                                h_data['waku'] = ((u_num - 1) // 2) + 1
                               
                            if m_pop:
                                h_data['popularity'] = int(m_pop.group(1))
                            if m_odds:
                                try:
                                    val = float(m_odds.group(1))
                                    if val < 1000:
                                        h_data['odds'] = val
                                except:
                                    pass
                            if m_sex_age:
                                s = m_sex_age.group(1)
                                if s == 'セ': s = '騸'
                                h_data['sex'] = s
                                h_data['age'] = int(m_sex_age.group(2))
                            if m_wt:
                                h_data['weight'] = float(m_wt.group(1))
                               
                            # 騎手名
                            jock_list = ['川田将雅', 'ルメール', '武豊', '戸崎圭太', '岩田望来', '北村友一', '池添謙一', '松山弘平', '藤懸貴志', '吉村誠之', '津村明秀', 'レーン', '松本大輝', '松若風馬', '鮫島克駿']
                            for j in jock_list:
                                if j in text_all:
                                    h_data['jockey'] = j
                                    break
                                   
                            parsed_horses.append(h_data)

                    if parsed_horses:
                        st.session_state['parsed_horses'] = parsed_horses
                        st.success(f"✨ URLから {len(parsed_horses)}頭分のデータを自動取得しました！")
                    else:
                        st.warning("⚠️ 該当ページからデータを抽出できませんでした。URLが正しいか確認してください。")
                except Exception as e:
                    st.error(f"⚠️ 取得エラー: {e}")
            else:
                st.warning("⚠️ URLを入力してください。")

    parsed_data_list = st.session_state.get('parsed_horses', [])
    default_num = max(len(parsed_data_list), 8)

    num_horses = st.slider("予測する出馬頭数", min_value=1, max_value=18, value=default_num, key="pred_num")
   
    input_data_list = []

    if model is None:
        st.error("⚠️ AIモデルが見つかりません。「ガチAIの再学習」タブからモデルを作成してください。")
    else:
        st.success("✨ ガチAIモデル稼働中！")

    for i in range(num_horses):
        auto_umaban = i + 1
        auto_waku = ((auto_umaban - 1) // 2) + 1
        if auto_waku > 8: auto_waku = 8

        def_name = ""
        def_sex = "牡"
        def_age = 3
        def_weight = 56.0
        def_jockey = ""
        def_sire = ""
        def_odds = 0.0
        def_pop = 0

        if i < len(parsed_data_list):
            h_info = parsed_data_list[i]
            if isinstance(h_info, dict):
                auto_umaban = h_info.get('umaban', auto_umaban)
                auto_waku = h_info.get('waku', auto_waku)
                def_name = h_info.get('name', '')
                def_sex = h_info.get('sex', '牡')
                def_age = h_info.get('age', 3)
                def_weight = h_info.get('weight', 56.0)
                def_sire = h_info.get('sire', '')
                def_odds = h_info.get('odds', 0.0)
                def_pop = h_info.get('popularity', 0)
                def_jockey = h_info.get('jockey', '')

        with st.expander(f"馬番 {auto_umaban}: {def_name if def_name else '未設定馬'}", expanded=(i < 3)):
            col1, col2 = st.columns(2)
            with col1:
                horse_name = st.text_input(f"馬名", value=def_name, key=f"p_name_{i}")
                r_sex = st.selectbox(f"性別", ["牡", "牝", "騸"], index=["牡", "牝", "騸"].index(def_sex) if def_sex in ["牡", "牝", "騸"] else 0, key=f"p_sex_{i}")
                r_age = st.number_input(f"年齢", min_value=2, max_value=15, value=def_age, key=f"p_age_{i}")
                weight = st.number_input(f"斤量", value=def_weight, step=0.5, key=f"p_weight_{i}")
                r_sire = st.text_input(f"父馬名", value=def_sire, key=f"p_sire_{i}")
            with col2:
                odds = st.number_input(f"単勝オッズ", value=def_odds, min_value=0.0, step=0.1, key=f"p_odds_{i}")
                popularity = st.number_input(f"人気順", value=def_pop, min_value=0, step=1, key=f"p_pop_{i}")
                r_jockey = st.text_input(f"騎手名", value=def_jockey, key=f"p_jockey_{i}")
                r_blinker = st.selectbox(f"ブリンカー", ["", "B"], index=0, key=f"p_blinker_{i}")
           
            jock_rate = jockey_win_rates.get(r_jockey, 0.08)
            is_blinker = 1 if r_blinker == "B" else 0

            input_data_list.append({
                'place': p_place,
                'track': p_track,
                'distance': p_distance,
                'condition': p_condition,
                'waku': auto_waku,
                'umaban': auto_umaban,
                'name': horse_name if horse_name else f"馬番{auto_umaban}",
                'sex': r_sex,
                'age': r_age,
                'sire': r_sire if r_sire else "不明",
                'odds': odds if odds > 0 else 10.0,
                'popularity': popularity if popularity > 0 else auto_umaban,
                'weight': weight,
                'jockey': r_jockey if r_jockey else "不明",
                'jockey_win_rate': jock_rate,
                'blinker': is_blinker
            })

    if model is not None and st.button("🚀 ガチ予測を実行する！"):
        df_input = pd.DataFrame(input_data_list)
       
        if len(df_m_auto) > 0:
            df_full = pd.concat([df_m_auto, df_input], ignore_index=True)
        else:
            df_full = df_input

        df_full = df_full.loc[:, ~df_full.columns.duplicated()]

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
            st.warning("⚠️ エラーが発生しました:")
            st.write(e)

with tab2:
    st.subheader("📝 レース結果をマスターに追加する")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        race_date = st.date_input("開催日", datetime.date(2026, 6, 1))
        race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
        race_number = st.number_input("レース番号 (R)", min_value=1, max_value=12, value=11, key="r_num")
    with col_r2:
        race_name = st.text_input("レース名", "", placeholder="例: 日本ダービー", key="r_name_input")
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
                r_name = st.text_input(f"馬名", "", key=f"r_name_{i}")
                r_sex = st.selectbox(f"性別", ["牡", "牝", "騸"], key=f"r_sex_{i}")
                r_age = st.number_input(f"年齢", min_value=2, max_value=15, value=3, key=f"r_age_{i}")
                r_jockey = st.text_input(f"騎手名", "", key=f"r_jockey_{i}")
                r_trainer = st.text_input(f"調教師名", "", key=f"r_trainer_{i}")
                r_stable = st.selectbox(f"所属", ["美浦", "栗東", "地方", "海外"], key=f"r_stable_{i}")
            with col_b:
                r_sire = st.text_input(f"父馬名", "", key=f"r_sire_{i}")
                r_dam = st.text_input(f"母馬名", "", key=f"r_dam_{i}")
                r_weight = st.number_input(f"斤量", value=56.0, step=0.5, key=f"r_weight_{i}")
                r_blinker_res = st.selectbox(f"ブリンカー", ["", "B"], key=f"r_blinker_{i}")
                r_rank = st.number_input(f"確定着順", min_value=1, max_value=18, value=1, key=f"r_rank_{i}")
                r_odds = st.number_input(f"単勝オッズ", value=0.0, min_value=0.0, step=0.1, key=f"r_odds_{i}")
                r_pop = st.number_input(f"人気順", value=0, min_value=0, step=1, key=f"r_pop_{i}")

            new_data_list.append({
                'date': race_date,
                'place': race_place,
                'race_number': race_number,
                'race_name': race_name if race_name else "レース名",
                'track': track_type,
                'distance': distance,
                'condition': condition,
                'weather': weather,
                'waku': auto_waku,
                'umaban': auto_umaban,
                'horse_name': r_name if r_name else f"馬番{auto_umaban}",
                'sex': r_sex,
                'age': r_age,
                'jockey': r_jockey if r_jockey else "不明",
                'trainer': r_trainer if r_trainer else "不明",
                'stable': r_stable,
                'sire': r_sire if r_sire else "不明",
                'dam': r_dam if r_dam else "不明",
                'weight': r_weight,
                'blinker': r_blinker_res,
                'rank': r_rank,
                'odds': r_odds if r_odds > 0 else 1.0,
                'popularity': r_pop if r_pop > 0 else 1
            })

    MASTER_FILENAME = 'keiba_master_data.csv'
    if st.button("🚀 追加データをマスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        if os.path.exists(MASTER_FILENAME):
            try:
                df_master = pd.read_csv(MASTER_FILENAME, encoding='cp932', low_memory=False)
            except Exception:
                df_master = pd.read_csv(MASTER_FILENAME, encoding='utf-8-sig', low_memory=False)
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(MASTER_FILENAME, index=False, encoding='cp932')
        st.balloons()
        st.success(f"🎉 追加成功！マスターデータに結果が保存されました！✨")

with tab3:
    st.subheader("🧠 全データでガチAIを再学習させる")
   
    if len(df_m_auto) > 10:
        st.success(f"📂 合算データ読み込み成功！学習データ総数: **{len(df_m_auto):,} 行**")

        if st.button("🚀 フルデータでAIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
               
                target_train_df = df_m_auto.copy()
                target_train_df = target_train_df.loc[:, ~target_train_df.columns.duplicated()]
               
                if len(target_train_df) > 50000:
                    target_train_df = target_train_df.sample(n=50000, random_state=42)
                   
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

                if 'rank' in target_train_df.columns:
                    target_train_df['target'] = (target_train_df['rank'] == 1).astype(int)
                else:
                    st.error("⚠️ 'rank'（着順）の列が見つかりません。")
                    st.stop()
               
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker']
                for feat in features:
                    if feat not in target_train_df.columns:
                        target_train_df[feat] = 0
               
                X = target_train_df[features].fillna(0)
                y = target_train_df['target']

                clf = lgb.LGBMClassifier(random_state=42)
                clf.fit(X, y)

                joblib.dump(clf, 'keiba_ai_model.pkl')
                st.balloons()
                st.success("🎉 再学習がバッチリ完了しました！✨")

            except Exception as e:
                st.warning(f"⚠️ 学習エラー: {e}")
    else:
        st.warning("⚠️ 十分なデータが読み込めていません。")
