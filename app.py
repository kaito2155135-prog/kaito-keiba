import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import unicodedata
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

st.title("🐎【上がり3F特化・最強網羅版】スマホで育てる！競馬AIマスターアプリ")
st.write("直近データ全行 ＆ 上がり3F分析エンジン・展開逆行ギャップ算出稼働中！✨🔥")

def clean_str(s):
    if not s:
        return ""
    return unicodedata.normalize('NFKC', str(s)).replace(" ", "").replace("　", "").strip()

def load_model():
    try:
        if os.path.exists('keiba_ai_model.pkl'):
            return joblib.load('keiba_ai_model.pkl')
    except Exception:
        return None
    return None

model = load_model()

def parse_corner_positions_from_row(row):
    c1, c4 = np.nan, np.nan
    corner_vals = []
    
    for col in ['通過順1角', '通過順1', '1角', '通過順2角', '通過順2', '2角', '通過順3角', '通過順3', '3角', '通過順4角', '通過順4', '4角']:
        if col in row and pd.notna(row[col]):
            try:
                val = float(str(row[col]).strip())
                if val > 0:
                    corner_vals.append((col, val))
            except:
                pass

    if not corner_vals and 'corner' in row and pd.notna(row['corner']):
        try:
            s = str(row['corner']).strip()
            for sep in [',', ' ', '/', '->', '－', '―']:
                s = s.replace(sep, '-')
            parts = [float(p.strip()) for p in s.split('-') if p.strip().replace('.', '', 1).isdigit()]
            corner_vals = [('str', p) for p in parts if p > 0]
        except:
            pass

    if corner_vals:
        c1 = corner_vals[0][1]
        c4 = corner_vals[-1][1]

    return c1, c4

def parse_corner_string_input(val):
    try:
        s = str(val).strip()
        if not s or s == 'nan':
            return np.nan, np.nan
        for sep in [',', ' ', '/', '->', '－', '―']:
            s = s.replace(sep, '-')
        parts = [float(p.strip()) for p in s.split('-') if p.strip().replace('.', '', 1).isdigit()]
        valid_parts = [p for p in parts if p > 0]
        if len(valid_parts) >= 2:
            return valid_parts[0], valid_parts[-1]
        elif len(valid_parts) == 1:
            return valid_parts[0], valid_parts[0]
    except:
        pass
    return np.nan, np.nan

def parse_time_to_sec(val):
    try:
        s = str(val).strip()
        if ':' in s:
            m, rest = s.split(':')
            return float(m) * 60 + float(rest)
        elif s and s != 'nan':
            return float(s)
    except:
        pass
    return 0.0

@st.cache_data
def load_and_process_master_data():
    filename = 'keiba_master_data_part2.csv'
    if not os.path.exists(filename):
        for alt in ['keiba_master_data.csv', 'keiba_master_data_part1.csv']:
            if os.path.exists(alt):
                filename = alt
                break
        else:
            return pd.DataFrame(), {}, {}, {}

    df_m = pd.DataFrame()
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df_m = pd.read_csv(filename, encoding=enc, low_memory=False)
            if not df_m.empty:
                df_m.columns = [str(c).strip() for c in df_m.columns]

                col_mapping = {}
                for c in df_m.columns:
                    clean_c = c.replace(" ", "").replace("　", "")
                    if clean_c in ['馬名', 'horsename', 'H_Name', '競走馬名']: col_mapping[c] = 'name'
                    elif clean_c in ['着順', '順位', '確定着順', 'Rank', 'RANK']: col_mapping[c] = 'rank'
                    elif clean_c in ['タイム', 'Time', 'TIME', '走破タイム', '走破時間']: col_mapping[c] = 'time'
                    elif clean_c in ['騎手', 'Jockey', 'jockey']: col_mapping[c] = 'jockey'
                    elif clean_c in ['上がり3F', '上り3F', 'last_3f', 'L3F', '上がり', '上り', '上がり3Fタイム']: col_mapping[c] = 'last_3f'
                    elif clean_c in ['距離', 'Distance', 'distance']: col_mapping[c] = 'distance'
                    elif clean_c in ['馬場', '馬場状態', 'Condition', 'condition']: col_mapping[c] = 'condition'
                    elif clean_c in ['性別', 'Sex', 'sex']: col_mapping[c] = 'sex'
                    # ▼ ここを修正！ 列名が「芝・ダート」「芝」「ダート」でも track_type として認識させる
                    elif clean_c in ['トラック', 'Track', 'track', '芝・ダート', '芝', 'ダート', 'コース', 'トラック種別']: col_mapping[c] = 'track_type'
                
                df_m = df_m.rename(columns=col_mapping)
                break
        except Exception:
            continue

    if df_m.empty:
        return pd.DataFrame(), {}, {}, {}

    if 'name' in df_m.columns:
        df_m['name'] = df_m['name'].astype(str).apply(clean_str)
    if 'sex' in df_m.columns:
        df_m['sex'] = df_m['sex'].astype(str).str.strip()
    else:
        df_m['sex'] = '牡'

    if 'rank' in df_m.columns:
        df_m['rank'] = pd.to_numeric(df_m['rank'], errors='coerce').fillna(1)
    else:
        df_m['rank'] = 1

    if 'last_3f' in df_m.columns:
        df_m['last_3f'] = pd.to_numeric(df_m['last_3f'], errors='coerce')
        df_m.loc[(df_m['last_3f'] < 25.0) | (df_m['last_3f'] > 50.0), 'last_3f'] = np.nan
        df_m['last_3f'] = df_m['last_3f'].fillna(35.0)
    else:
        df_m['last_3f'] = 35.0

    if 'distance' in df_m.columns:
        df_m['distance'] = pd.to_numeric(df_m['distance'], errors='coerce').fillna(2000.0)
    else:
        df_m['distance'] = 2000.0

    parsed_corners = df_m.apply(parse_corner_positions_from_row, axis=1)
    df_m['corner_1st'] = [x[0] for x in parsed_corners]
    df_m['corner_4th'] = [x[1] for x in parsed_corners]
    df_m['gap'] = df_m['corner_4th'] - df_m['corner_1st']
    df_m['time_sec'] = df_m['time'].apply(parse_time_to_sec) if 'time' in df_m.columns else 0.0

    jockey_win_rates = {}
    if 'jockey' in df_m.columns and 'rank' in df_m.columns:
        j_stats = df_m.groupby('jockey').agg(total=('rank', 'count'), wins=('rank', lambda x: (x == 1).sum()))
        for j, row in j_stats.iterrows():
            if row['total'] > 0: jockey_win_rates[j] = row['wins'] / row['total']

    horse_history_features = {}
    horse_distance_features = {}

    if 'name' in df_m.columns:
        has_track_col = 'track_type' in df_m.columns
        h_grouped = df_m.groupby('name').agg(
            avg_rank=('rank', 'mean'),
            best_rank=('rank', 'min'),
            avg_time=('time_sec', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0.0),
            avg_last_3f=('last_3f', 'mean'),
            avg_corner_1st=('corner_1st', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
            avg_corner_4th=('corner_4th', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
            avg_gap=('gap', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
            race_count=('rank', 'count'),
            sex=('sex', lambda x: x.iloc[0] if len(x) > 0 and pd.notna(x.iloc[0]) else '牡'),
            # ▼ ここを修正！「ダート」という文字が含まれていれば経験ありとする
            has_dirt_exp=('track_type', lambda x: any('ダート' in str(t) for t in x if pd.notna(t))) if has_track_col else ('rank', lambda x: False)
        )
        
        for h_name, row in h_grouped.iterrows():
            c_name = clean_str(h_name)
            if c_name:
                horse_history_features[c_name] = {
                    'avg_rank': row['avg_rank'] if not np.isnan(row['avg_rank']) else 5.0,
                    'best_rank': row['best_rank'] if not np.isnan(row['best_rank']) else 5.0,
                    'avg_time': row['avg_time'] if not np.isnan(row['avg_time']) else 0.0,
                    'avg_last_3f': row['avg_last_3f'] if not np.isnan(row['avg_last_3f']) else 35.0,
                    'avg_corner_1st': row['avg_corner_1st'],
                    'avg_corner_4th': row['avg_corner_4th'],
                    'avg_gap': row['avg_gap'],
                    'race_count': row['race_count'],
                    'sex': row['sex'] if str(row['sex']).strip() in ['牡', '牝', 'セン', 'セ'] else '牡',
                    'has_dirt_exp': bool(row['has_dirt_exp']) if has_track_col else False
                }

        if 'distance' in df_m.columns:
            d_grouped = df_m.groupby(['name', 'distance']).agg(
                avg_rank=('rank', 'mean'),
                avg_last_3f=('last_3f', 'mean'),
                avg_corner_1st=('corner_1st', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
                avg_corner_4th=('corner_4th', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
                avg_gap=('gap', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else np.nan),
                race_count=('rank', 'count')
            )
            for (h_name, dist), row in d_grouped.iterrows():
                c_name = clean_str(h_name)
                if c_name:
                    if c_name not in horse_distance_features:
                        horse_distance_features[c_name] = {}
                    horse_distance_features[c_name][float(dist)] = {
                        'avg_rank': row['avg_rank'] if not np.isnan(row['avg_rank']) else 5.0,
                        'avg_last_3f': row['avg_last_3f'] if not np.isnan(row['avg_last_3f']) else 35.0,
                        'avg_corner_1st': row['avg_corner_1st'],
                        'avg_corner_4th': row['avg_corner_4th'],
                        'avg_gap': row['avg_gap'],
                        'race_count': row['race_count']
                    }

    return df_m, jockey_win_rates, horse_history_features, horse_distance_features

df_m_auto, jockey_win_rates, horse_history_features, horse_distance_features = load_and_process_master_data()

tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果を追加", "🧠 AI再学習"])

with tab1:
    st.subheader("🚀 勝ち馬のガチ予測（上がり3F＆展開逆行ギャップ反映版）")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        p_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="p_place")
        p_class = st.selectbox("クラス", ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"], key="p_class")
    with col_p2:
        p_track = st.selectbox("トラック", ["芝", "ダート", "障害"], key="p_track")
        p_distance = st.number_input("距離 (m)", value=2000, step=100, key="p_distance")
    with col_p3:
        p_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="p_condition")

    st.markdown("---")
    st.markdown("### 📋 出馬表テキストの一発ペースト")
    raw_text = st.text_area("ここにnetkeibaの出馬表をペースト", height=150, key="raw_text_input")

    input_data_list = []
    try:
        if raw_text.strip():
            lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip() != ""]
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.isdigit() and 1 <= int(line) <= 18:
                    umaban = int(line)
                    waku = min(8, max(1, ((umaban - 1) // 2) + 1))
                    h_name = f"馬番{umaban}"
                    sex = "牡"
                    age = 4
                    jockey = "不明"
                    weight = 56.0
                    odds = 10.0
                    popularity = umaban

                    block_lines = []
                    j = i + 1
                    while j < len(lines):
                        if lines[j].isdigit() and 1 <= int(lines[j]) <= 18:
                            break
                        block_lines.append(lines[j])
                        j += 1

                    for bl in block_lines:
                        clean_bl = clean_str(bl)
                        if clean_bl in horse_history_features:
                            h_name = clean_bl
                            break

                    if h_name.startswith("馬番"):
                        for bl in block_lines:
                            clean_b = clean_str(bl.replace("--", ""))
                            if clean_b and not any(kw in clean_b for kw in ["人気", "データベース", "牡", "牝", "セ", "kg"]) and len(clean_b) >= 2:
                                h_name = clean_b
                                break

                    for bl in block_lines:
                        if "データベース" in bl:
                            parts = bl.split("のデータベース")
                            if len(parts) > 0:
                                left_part = parts[0].strip()
                                for prefix in ["牡", "牝", "セ", "セン"]:
                                    if left_part.startswith(prefix):
                                        sex = "牝" if prefix == "牝" else ("セン" if prefix in ["セ", "セン"] else "牡")
                                        left_part = left_part[len(prefix):].lstrip()
                                        if left_part and left_part[0].isdigit():
                                            age = int(left_part[0])
                                            left_part = left_part[1:].lstrip()
                                if left_part and (h_name.startswith("馬番") or len(left_part) > len(h_name)):
                                    h_name = left_part

                            if len(parts) > 1:
                                right_part = parts[1].strip()
                                right_tokens = right_part.split()
                                if len(right_tokens) >= 2:
                                    jockey = right_tokens[0]
                                    try:
                                        weight = float(right_tokens[1])
                                    except:
                                        pass
                                elif len(right_tokens) == 1:
                                    jockey = right_tokens[0]

                        if any(s in bl for s in ["牡", "牝", "セン", "セ"]) and len(bl) <= 6:
                            if "牝" in bl: sex = "牝"
                            elif "セ" in bl or "セン" in bl: sex = "セン"
                            else: sex = "牡"
                            for char in bl:
                                if char.isdigit():
                                    age = int(char)
                                    break

                        if "人気" in bl:
                            try:
                                popularity = int(bl.replace("人気", "").strip())
                            except:
                                pass

                        if "." in bl and not any(kw in bl for kw in ["人気", "kg", "("]) and len(bl) <= 6:
                            try:
                                val = float(bl)
                                if 0.1 <= val < 3000:
                                    odds = val
                            except:
                                pass

                    clean_h_name = clean_str(h_name)

                    target_key = None
                    if clean_h_name in horse_history_features:
                        target_key = clean_h_name
                    else:
                        for m_name in horse_history_features.keys():
                            if clean_h_name in m_name or m_name in clean_h_name:
                                target_key = m_name
                                break

                    matched_hist = {
                        'avg_rank': 5.0, 'best_rank': 5.0, 'avg_time': 0.0, 'avg_last_3f': 35.0,
                        'avg_corner_1st': np.nan, 'avg_corner_4th': np.nan, 'avg_gap': np.nan,
                        'race_count': 0, 'sex': sex, 'has_dirt_exp': False
                    }
                    
                    if target_key and target_key in horse_history_features:
                        matched_hist = horse_history_features[target_key].copy()
                        clean_h_name = target_key

                    if matched_hist.get('sex') in ['牡', '牝', 'セン', 'セ']:
                        sex = matched_hist['sex']

                    if clean_h_name in horse_distance_features:
                        d_dict = horse_distance_features[clean_h_name]
                        dist_val = float(p_distance)
                        if dist_val in d_dict:
                            matched_hist.update(d_dict[dist_val])
                        else:
                            closest_dist = min(d_dict.keys(), key=lambda x: abs(x - dist_val))
                            if abs(closest_dist - dist_val) <= 400:
                                matched_hist.update(d_dict[closest_dist])

                    c1_val = matched_hist.get('avg_corner_1st', np.nan)
                    c4_val = matched_hist.get('avg_corner_4th', np.nan)
                    gap_val = matched_hist.get('avg_gap', np.nan)

                    is_short_distance = float(p_distance) <= 1200
                    if not is_short_distance:
                        if pd.isna(c1_val): c1_val = 5.0
                    
                    if pd.isna(c4_val): c4_val = 7.0
                    
                    if is_short_distance or pd.isna(c1_val):
                        gap_val = 0.0 if is_short_distance else (c4_val - 5.0)
                    else:
                        if pd.isna(gap_val): gap_val = c4_val - c1_val

                    input_data_list.append({
                        'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                        'race_class': p_class, 'waku': waku, 'umaban': umaban, 'name': clean_h_name, 'sex': sex, 'age': age, 'sire': '不明',
                        'odds': odds, 'popularity': popularity, 'weight': weight, 'jockey': jockey,
                        'jockey_win_rate': jockey_win_rates.get(jockey, 0.08),
                        'past_avg_rank': matched_hist['avg_rank'],
                        'past_best_rank': matched_hist['best_rank'],
                        'time_sec': matched_hist['avg_time'] if matched_hist['avg_time'] > 0 else 0.0,
                        'last_3f': matched_hist['avg_last_3f'],
                        'blinker': 0, 
                        'corner_1st': c1_val if not is_short_distance else np.nan,
                        'corner_4th': c4_val,
                        'gap': gap_val,
                        'has_dirt_exp': matched_hist.get('has_dirt_exp', False)
                    })
                    i = j - 1
                i += 1
    except Exception as e:
        input_data_list = []

    if len(input_data_list) == 0:
        st.warning("⚠️ テキスト未入力または解析対象外のため、デフォルトの8頭で表示しています。")
        np.random.seed(42)
        is_short_distance = float(p_distance) <= 1200
        for i in range(8):
            c1 = np.nan if is_short_distance else float(np.random.randint(1, 10))
            c4 = float(np.random.randint(1, 10))
            input_data_list.append({
                'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                'race_class': p_class, 'waku': 1, 'umaban': i+1, 'name': f"馬番{i+1}", 'sex': '牡', 'age': 4, 'sire': '不明',
                'odds': float(i+2), 'popularity': i+1, 'weight': 56.0, 'jockey': '不明', 'jockey_win_rate': 0.08,
                'past_avg_rank': 5.0, 'past_best_rank': 5.0, 'time_sec': 0.0, 'last_3f': 35.0, 'blinker': 0, 
                'corner_1st': c1, 'corner_4th': c4, 'gap': 0.0 if is_short_distance else (c4 - c1),
                'has_dirt_exp': False
            })
    else:
        matched_count = sum(1 for x in input_data_list if x['past_avg_rank'] != 5.0)
        st.success(f"✨ テキストから出走馬 **{len(input_data_list)}頭** を検出！(うち直近データ一致: **{matched_count}頭** / Part2総読込行数: {len(df_m_auto):,})")

    if st.button("🚀 ガチ予測を実行する！"):
        df_input = pd.DataFrame(input_data_list)
        df_input['odds'] = pd.to_numeric(df_input['odds'], errors='coerce').fillna(10.0)
        df_input['popularity'] = pd.to_numeric(df_input['popularity'], errors='coerce').fillna(99)
        df_input['distance'] = float(p_distance)
        df_input['condition'] = str(p_condition)

        if model is not None:
            try:
                df_full = pd.concat([df_m_auto, df_input], ignore_index=True) if not df_m_auto.empty else df_input
                df_full = df_full.loc[:, ~df_full.columns.duplicated()]
                for col in ['place', 'track', 'condition', 'sire', 'race_class']:
                    if col in df_full.columns:
                        df_full[col] = LabelEncoder().fit_transform(df_full[col].astype(str))
                df_input_enc = df_full.tail(len(df_input)).copy()
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker', 'corner_1st', 'corner_4th', 'gap', 'race_class', 'time_sec', 'last_3f']
                for f in features:
                    if f not in df_input_enc.columns: df_input_enc[f] = 0
                model_probs = model.predict_proba(df_input_enc[features].fillna(0))[:, 1]

                l3f_bonus = (38.0 - df_input['last_3f']).clip(lower=0) * 2.0
                score = model_probs * 3.0 + (6.0 - df_input['past_avg_rank']).clip(lower=0) * 1.5 + l3f_bonus + df_input['jockey_win_rate'] * 2.0 + (1.0 / np.log1p(df_input['odds'])) * 1.5
            except Exception as e:
                l3f_bonus = (38.0 - df_input['last_3f']).clip(lower=0) * 2.0
                score = (6.0 - df_input['past_avg_rank']).clip(lower=0) * 2.0 + l3f_bonus + df_input['jockey_win_rate'] * 2.0 + (1.0 / np.log1p(df_input['odds'])) * 1.5
        else:
            l3f_bonus = (38.0 - df_input['last_3f']).clip(lower=0) * 2.0
            score = (6.0 - df_input['past_avg_rank']).clip(lower=0) * 2.0 + l3f_bonus + df_input['jockey_win_rate'] * 2.0 + (1.0 / np.log1p(df_input['odds'])) * 1.5

        if p_track == "ダート":
            for idx, row in df_input.iterrows():
                if not row.get('has_dirt_exp', False):
                    score.iloc[idx] *= 0.6  

        exp_s = np.exp(score - score.max())
        df_input['win_prob'] = (exp_s / exp_s.sum()) * 100

        df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
        st.balloons()
        st.subheader("🎯 ガチAI予測結果ランキング（上がり3F＆展開逆行ギャップ反映版）")

        for idx, row in df_input.iterrows():
            u_num = row.get('umaban', idx+1)
            h_sex = row.get('sex', '牡')
            h_age = row.get('age', 4)
            h_name = row.get('name', f'馬番{u_num}')
            h_prob = row.get('win_prob', 0.0)
            h_avg = row.get('past_avg_rank', 5.0)
            h_l3f = row.get('last_3f', 35.0)
            h_odds = row.get('odds', 10.0)
            h_jockey = row.get('jockey', '不明')
            c_1st = row.get('corner_1st', np.nan)
            c_4th = row.get('corner_4th', 7.0)
            c_gap = row.get('gap', 0.0)
            h_dirt_exp = row.get('has_dirt_exp', False)

            is_short = float(p_distance) <= 1200 or pd.isna(c_1st)
            corner_str = f"4角: {c_4th:.1f}" if is_short else f"通過順[1角->4角]: {c_1st:.1f}->{c_4th:.1f}"
            dirt_tag = " ⚠️(初ダート)" if p_track == "ダート" and not h_dirt_exp else ""

            st.write(f"**第 {idx+1} 位**: 馬番 {u_num} 🐴 {h_sex}{h_age} **{h_name}**{dirt_tag} (予測勝率: **{h_prob:.2f}%** / 直近平均着順: {h_avg:.1f}着 / {corner_str} / 展開逆行ギャップ: {c_gap:+.1f} / 上がり3F: {h_l3f:.1f}秒 / オッズ: {h_odds}倍 / 騎手: {h_jockey})")

with tab2:
    st.subheader("📝 レース結果をPart2マスターに追加する（上がり3F＆通過順入力対応）")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1: r_year = st.number_input("年", min_value=2000, max_value=2030, value=2026, key="r_year")
    with col_d2: r_month = st.number_input("月", min_value=1, max_value=12, value=6, key="r_month")
    with col_d3: r_day = st.number_input("日", min_value=1, max_value=31, value=1, key="r_day")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
        race_class = st.selectbox("クラス", ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"], key="r_class")
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
            col_a, col_b = st.columns(2)
            with col_a:
                r_name = st.text_input("馬名", f"馬番{u_num}", key=f"r_name_{i}")
                r_rank = st.number_input("確定着順", min_value=1, max_value=18, value=1, key=f"r_rank_{i}")
                r_odds = st.number_input("単勝オッズ", value=10.0, min_value=0.0, step=0.1, key=f"r_odds_{i}")
                r_pop = st.number_input("人気順", value=u_num, min_value=1, step=1, key=f"r_pop_{i}")
                r_blinker_str = st.selectbox("ブリンカー", ["なし", "B (あり)"], key=f"r_blinker_{i}")
                r_blinker = 1 if "B" in r_blinker_str else 0

            with col_b:
                r_jockey = st.text_input("騎手名", "不明", key=f"r_jockey_{i}")
                r_weight = st.number_input("斤量", value=56.0, step=0.5, key=f"r_weight_{i}")
                r_sire = st.text_input("父馬名", "不明", key=f"r_sire_{i}")
                r_age = st.number_input("年齢", min_value=2, max_value=15, value=4, key=f"r_age_{i}")
                r_corner = st.text_input("通過順 (例: 5-7-8-10)", "5-5-4-3", key=f"r_corner_{i}")
                r_time = st.text_input("走破タイム (例: 2:00.0)", "2:00.0", key=f"r_time_{i}")
                r_l3f = st.number_input("上がり3Fタイム (例: 34.5)", min_value=25.0, max_value=50.0, value=35.0, step=0.1, key=f"r_l3f_{i}")

            c_1st_val, c_4th_val = parse_corner_string_input(r_corner)
            gap_val = c_4th_val - c_1st_val if not pd.isna(c_1st_val) and not pd.isna(c_4th_val) else 0.0

            new_data_list.append({
                'year': r_year, 'month': r_month, 'day': r_day,
                'place': race_place, 'track': track_type, 'distance': distance, 'condition': condition,
                'race_class': race_class, 'waku': ((u_num-1)//2)+1, 'umaban': u_num, 'name': clean_str(r_name),
                'sex': '牡', 'age': r_age, 'jockey': r_jockey, 'sire': r_sire, 'weight': r_weight,
                'rank': r_rank, 'odds': r_odds, 'popularity': r_pop, 'blinker': r_blinker,
                'corner': r_corner, '通過順1角': c_1st_val, '通過順2角': c_4th_val, 'corner_1st': c_1st_val, 'corner_4th': c_4th_val, 'gap': gap_val,
                'time': r_time, 'time_sec': parse_time_to_sec(r_time), 'last_3f': r_l3f, 'track_type': track_type
            })

    if st.button("🚀 追加データをPart2マスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        df_combined = pd.concat([df_m_auto, df_new], ignore_index=True) if not df_m_auto.empty else df_new
        df_combined.to_csv('keiba_master_data_part2.csv', index=False, encoding='cp932')
        st.balloons()
        st.success("🎉 通過順（1角→4角）及び展開逆行ギャップ、上がり3Fデータを含む結果がPart2マスターに保存されました！")

with tab3:
    st.subheader("🧠 ガチAIを再学習させる")
    if not df_m_auto.empty:
        st.success(f"📂 Part2マスターデータ読み込み成功！ (読込行数: {len(df_m_auto):,})")
        if st.button("🚀 AIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
                df_train = df_m_auto.copy().loc[:, ~df_m_auto.columns.duplicated()]
                for col_name, default_val in [('rank', 1), ('last_3f', 35.0), ('distance', 2000.0), ('condition', '良'), ('corner_1st', 5.0), ('corner_4th', 7.0), ('gap', 2.0)]:
                    if col_name not in df_train.columns:
                        df_train[col_name] = default_val

                if len(df_train) > 10000: df_train = df_train.sample(n=10000, random_state=42)
                df_train['target'] = (pd.to_numeric(df_train['rank'], errors='coerce') == 1).astype(int)

                for col in ['place', 'track', 'condition', 'sire', 'race_class']:
                    if col in df_train.columns:
                        df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))

                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker', 'corner_1st', 'corner_4th', 'gap', 'race_class', 'time_sec', 'last_3f']
                for f in features:
                    if f not in df_train.columns: df_train[f] = 0

                X = df_train[features].fillna(0)
                y = df_train['target']
                clf = lgb.LGBMClassifier(random_state=42)
                clf.fit(X, y)
                joblib.dump(clf, 'keiba_ai_model.pkl')
                st.balloons()
                st.success("🎉 再学習完了！展開逆行ギャップや上がり3Fのファクターも含めてモデルがアップデートされました！")
            except Exception as e:
                st.warning(f"⚠️ 学習エラー: {e}")
    else:
        st.warning("⚠️ Part2マスターデータが見つかりません。")
