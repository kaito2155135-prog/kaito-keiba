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

st.title("🐎【昇級初戦・初ダート・初芝の壁・自動補正付き】スマホで育てる！競馬AIマスターアプリ")
st.write("ベースは直近7走平均＆競馬場トラック一致！さらに『昇級初戦・初ダート・初芝の過剰人気馬が凡走する人間の感覚』をAIのスコアに自動反映する進化版！🔥")

def clean_str(s):
    if not s:
        return ""
    res = unicodedata.normalize('NFKC', str(s))
    for c in [" ", "　", "\t", "\n", "\r", "・", "."]:
        res = res.replace(c, "")
    return res.strip()

def load_model():
    try:
        if os.path.exists('keiba_ai_model.pkl'):
            return joblib.load('keiba_ai_model.pkl')
    except Exception:
        return None
    return None

model = load_model()

# ★クラスの序列マップ（オープン以上と重賞をそれぞれ一括り）
CLASS_RANK_MAP = {
    "新馬": 0, 
    "未勝利": 1, 
    "1勝": 2, "1勝クラス": 2, 
    "2勝": 3, "2勝クラス": 3, 
    "3勝": 4, "3勝クラス": 4, 
    "オープン": 5, "OP": 5, "OP(L)": 5, "リステッド": 5,
    "G3": 6, "G2": 6, "G1": 6
}

def get_class_level(class_name):
    c_str = clean_str(str(class_name))
    for k, v in CLASS_RANK_MAP.items():
        if k in c_str:
            return v
    return 2 # デフォルトは1勝クラス相当

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
def load_master_data():
    filename = 'keiba_master_data_part2.csv'
    if not os.path.exists(filename):
        for alt in ['keiba_master_data.csv', 'keiba_master_data_part1.csv']:
            if os.path.exists(alt):
                filename = alt
                break
        else:
            return pd.DataFrame(), {}

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
                    elif clean_c in ['トラック', 'Track', 'track', 'コース', 'トラック種別']: col_mapping[c] = 'track'
                    elif clean_c in ['開催', '場所', '競馬場', 'Place', 'place']: col_mapping[c] = 'place'
                    elif clean_c in ['レースID', 'race_id', 'RaceID', 'R_ID']: col_mapping[c] = 'race_id'
                    elif clean_c in ['クラス', 'レースクラス', 'Class', 'race_class']: col_mapping[c] = 'race_class'
                
                df_m = df_m.rename(columns=col_mapping)
                break
        except Exception:
            continue

    if df_m.empty:
        return pd.DataFrame(), {}

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

    if 'place' in df_m.columns:
        df_m['place'] = df_m['place'].astype(str).apply(clean_str)
    else:
        df_m['place'] = '東京'

    if 'track' in df_m.columns:
        df_m['track'] = df_m['track'].astype(str).str.strip()
    else:
        df_m['track'] = '芝'

    if 'race_class' not in df_m.columns:
        df_m['race_class'] = '1勝クラス'

    jockey_win_rates = {}
    if 'jockey' in df_m.columns and 'rank' in df_m.columns:
        j_stats = df_m.groupby('jockey').agg(total=('rank', 'count'), wins=('rank', lambda x: (x == 1).sum()))
        for j, row in j_stats.iterrows():
            if row['total'] > 0: jockey_win_rates[j] = row['wins'] / row['total']

    return df_m, jockey_win_rates

df_m_auto, jockey_win_rates = load_master_data()

tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果を追加", "🧠 AI再学習"])

with tab1:
    st.subheader("🚀 勝ち馬のガチ予測（昇級初戦・初ダート・初芝の壁・自動補正付き）")

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
    current_target_class_level = get_class_level(p_class)

    try:
        if raw_text.strip():
            lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip() != ""]
            
            pasted_horse_candidates = set()
            for line in lines:
                c_l = clean_str(line)
                if len(c_l) >= 2 and not any(kw in c_l for kw in ["人気", "データベース", "牡", "牝", "セ", "kg", "東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"]):
                    pasted_horse_candidates.add(c_l)

            sub_df_m = pd.DataFrame()
            if not df_m_auto.empty and 'name' in df_m_auto.columns:
                sub_df_m = df_m_auto[df_m_auto['name'].isin(pasted_horse_candidates)].copy()
                if not sub_df_m.empty:
                    parsed_corners = sub_df_m.apply(parse_corner_positions_from_row, axis=1)
                    sub_df_m['corner_1st'] = [x[0] for x in parsed_corners]
                    sub_df_m['corner_4th'] = [x[1] for x in parsed_corners]
                    sub_df_m['time_sec'] = sub_df_m['time'].apply(parse_time_to_sec) if 'time' in sub_df_m.columns else 0.0

                    if 'race_id' in sub_df_m.columns:
                        top3_df = sub_df_m[sub_df_m['rank'] <= 3]
                        race_bias = top3_df.groupby('race_id')['corner_4th'].mean().to_dict()
                        sub_df_m['race_bias_4th'] = sub_df_m['race_id'].map(race_bias).fillna(6.0)
                    else:
                        sub_df_m['race_bias_4th'] = 6.0
                    sub_df_m['true_reverse_gap'] = sub_df_m['corner_4th'] - sub_df_m['race_bias_4th']

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
                        if not sub_df_m.empty and clean_bl in sub_df_m['name'].values:
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

                    matched_hist = {
                        'avg_rank': 5.0, 'best_rank': 5.0, 'avg_time': 0.0, 'avg_last_3f': 35.0,
                        'avg_corner_1st': np.nan, 'avg_corner_4th': np.nan, 'avg_true_reverse_gap': 0.0,
                        'race_count': 0, 'sex': sex, 'has_dirt_exp': True, 'has_turf_exp': True,
                        'is_promotion_first': False, 'prev_class_name': '不明', 
                        'is_first_dirt': False, 'is_first_turf': False
                    }

                    if not sub_df_m.empty:
                        h_group = sub_df_m[sub_df_m['name'] == clean_h_name]
                        if h_group.empty:
                            for m_name in sub_df_m['name'].unique():
                                if len(clean_h_name) >= 3 and (clean_h_name in m_name or m_name in clean_h_name):
                                    h_group = sub_df_m[sub_df_m['name'] == m_name]
                                    clean_h_name = m_name
                                    break

                        if not h_group.empty:
                            sex_val = h_group['sex'].iloc[0] if pd.notna(h_group['sex'].iloc[0]) else '牡'
                            if str(sex_val).strip() in ['牡', '牝', 'セン', 'セ']:
                                sex = sex_val

                            has_dirt = h_group['track'].astype(str).str.contains('ダート|ダ', regex=True).any()
                            has_turf = h_group['track'].astype(str).str.contains('芝', regex=True).any()
                            race_cnt = len(h_group)

                            # ★初ダート判定：今回がダート戦で、過去にダート出走経験がない場合
                            if p_track == "ダート" and not has_dirt and race_cnt > 0:
                                matched_hist['is_first_dirt'] = True

                            # ★初芝判定：今回が芝戦で、過去に芝出走経験がない場合
                            if p_track == "芝" and not has_turf and race_cnt > 0:
                                matched_hist['is_first_turf'] = True

                            # ★直近のレースから前走のクラスを判定
                            if 'race_class' in h_group.columns and len(h_group) > 0:
                                last_row = h_group.iloc[-1]
                                prev_c_name = str(last_row.get('race_class', '1勝クラス'))
                                prev_level = get_class_level(prev_c_name)
                                matched_hist['prev_class_name'] = prev_c_name
                                
                                # 今回のクラスレベルが前走より高い ＝ 昇級初戦！
                                if current_target_class_level > prev_level:
                                    matched_hist['is_promotion_first'] = True

                            if p_track == "芝":
                                sub_sub = h_group[h_group['track'].apply(lambda x: '芝' in str(x) and 'ダ' not in str(x))]
                                if len(sub_sub) == 0: sub_sub = h_group
                            elif p_track == "ダート":
                                sub_sub = h_group[h_group['track'].apply(lambda x: 'ダ' in str(x))]
                                if len(sub_sub) == 0: sub_sub = h_group
                            else:
                                sub_sub = h_group

                            recent_sub = sub_sub.dropna(subset=['rank']).tail(7)
                            if len(recent_sub) == 0:
                                recent_sub = h_group.dropna(subset=['rank']).tail(7)

                            if len(recent_sub) > 0:
                                matched_hist['avg_rank'] = recent_sub['rank'].mean()
                                matched_hist['best_rank'] = h_group['rank'].min()
                                valid_times = recent_sub['time_sec'][recent_sub['time_sec'] > 0]
                                matched_hist['avg_time'] = valid_times.mean() if len(valid_times) > 0 else 0.0
                                matched_hist['avg_last_3f'] = recent_sub['last_3f'].mean() if len(recent_sub['last_3f'].dropna()) > 0 else 35.0
                                matched_hist['avg_corner_1st'] = recent_sub['corner_1st'].mean() if len(recent_sub['corner_1st'].dropna()) > 0 else np.nan
                                matched_hist['avg_corner_4th'] = recent_sub['corner_4th'].mean() if len(recent_sub['corner_4th'].dropna()) > 0 else np.nan
                                matched_hist['avg_true_reverse_gap'] = recent_sub['true_reverse_gap'].mean() if len(recent_sub['true_reverse_gap'].dropna()) > 0 else 0.0

                            matched_hist['race_count'] = race_cnt
                            matched_hist['has_dirt_exp'] = has_dirt
                            matched_hist['has_turf_exp'] = has_turf

                            exact_match = h_group[
                                (h_group['place'] == clean_str(p_place)) & 
                                (h_group['track'].astype(str).str.contains(p_track))
                            ]
                            if len(exact_match) > 0:
                                matched_hist['avg_rank'] = matched_hist['avg_rank'] * 0.5 + exact_match['rank'].mean() * 0.5
                                matched_hist['avg_last_3f'] = matched_hist['avg_last_3f'] * 0.5 + exact_match['last_3f'].mean() * 0.5
                                
                                c1_ex = exact_match['corner_1st'].mean()
                                if pd.notna(c1_ex):
                                    base_c1 = matched_hist.get('avg_corner_1st', c1_ex)
                                    matched_hist['avg_corner_1st'] = base_c1 * 0.5 + c1_ex * 0.5
                                
                                c4_ex = exact_match['corner_4th'].mean()
                                if pd.notna(c4_ex):
                                    base_c4 = matched_hist.get('avg_corner_4th', c4_ex)
                                    matched_hist['avg_corner_4th'] = base_c4 * 0.5 + c4_ex * 0.5

                                gap_ex = exact_match['true_reverse_gap'].mean()
                                if pd.notna(gap_ex):
                                    base_gap = matched_hist.get('avg_true_reverse_gap', gap_ex)
                                    matched_hist['avg_true_reverse_gap'] = base_gap * 0.5 + gap_ex * 0.5

                    c1_val = matched_hist.get('avg_corner_1st', np.nan)
                    c4_val = matched_hist.get('avg_corner_4th', np.nan)
                    true_gap = matched_hist.get('avg_true_reverse_gap', 0.0)

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
                        'corner_1st': c1_val,
                        'corner_4th': c4_val,
                        'true_reverse_gap': true_gap,
                        'has_dirt_exp': matched_hist.get('has_dirt_exp', True),
                        'has_turf_exp': matched_hist.get('has_turf_exp', True),
                        'has_history': matched_hist['race_count'] > 0,
                        'is_promotion_first': matched_hist['is_promotion_first'],
                        'prev_class_name': matched_hist['prev_class_name'],
                        'is_first_dirt': matched_hist['is_first_dirt'],
                        'is_first_turf': matched_hist['is_first_turf']
                    })
                    i = j - 1
                i += 1
    except Exception as e:
        input_data_list = []

    if len(input_data_list) == 0:
        st.warning("⚠️ テキスト未入力または解析対象外のため、デフォルトの8頭で表示しています。")
        np.random.seed(42)
        for i in range(8):
            input_data_list.append({
                'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
                'race_class': p_class, 'waku': 1, 'umaban': i+1, 'name': f"馬番{i+1}", 'sex': '牡', 'age': 4, 'sire': '不明',
                'odds': float(i+2), 'popularity': i+1, 'weight': 56.0, 'jockey': '不明', 'jockey_win_rate': 0.08,
                'past_avg_rank': 5.0, 'past_best_rank': 5.0, 'time_sec': 0.0, 'last_3f': 35.0, 'blinker': 0, 
                'corner_1st': 5.0, 'corner_4th': 7.0, 'true_reverse_gap': 0.0,
                'has_dirt_exp': True, 'has_turf_exp': True, 'has_history': False,
                'is_promotion_first': False, 'prev_class_name': '不明',
                'is_first_dirt': False, 'is_first_turf': False
            })
    else:
        matched_count = sum(1 for x in input_data_list if x['has_history'])
        promo_count = sum(1 for x in input_data_list if x['is_promotion_first'])
        dirt_count = sum(1 for x in input_data_list if x['is_first_dirt'])
        turf_count = sum(1 for x in input_data_list if x['is_first_turf'])
        st.success(f"✨ テキストから出走馬 **{len(input_data_list)}頭** を検出！(データ一致: {matched_count}頭 / ⚡️昇級初戦: {promo_count}頭 / ⚠️初ダート: {dirt_count}頭 / ⚠️初芝: **{turf_count}頭**) ")

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
                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker', 'corner_1st', 'corner_4th', 'true_reverse_gap', 'race_class', 'time_sec', 'last_3f']
                for f in features:
                    if f not in df_input_enc.columns: df_input_enc[f] = 0
                model_probs = model.predict_proba(df_input_enc[features].fillna(0))[:, 1]

                rank_score = (15.0 - df_input['past_avg_rank']).clip(lower=-10.0) * 4.0
                l3f_bonus = (38.0 - df_input['last_3f']).clip(lower=0) * 2.0
                is_capable = df_input['past_avg_rank'] <= 7.0
                reverse_val = df_input['true_reverse_gap'].abs()
                reverse_bonus = np.where(is_capable, reverse_val * 1.5, -reverse_val * 0.5)

                # 各種マイナス補正
                promo_penalty = np.where(df_input['is_promotion_first'], -5.0, 0.0)
                dirt_penalty = np.where(df_input['is_first_dirt'], -6.0, 0.0)
                turf_penalty = np.where(df_input['is_first_turf'], -6.0, 0.0)

                score = model_probs * 2.5 + rank_score + l3f_bonus + reverse_bonus + promo_penalty + dirt_penalty + turf_penalty + df_input['jockey_win_rate'] * 2.0 + (1.0 / np.log1p(df_input['odds'])) * 1.5
            except Exception as e:
                rank_score = (15.0 - df_input['past_avg_rank']).clip(lower=-10.0) * 4.0
                l3f_bonus = (38.0 - df_input['last_3f']).clip(lower=0) * 2.0
                is_capable = df_input['past_avg_rank'] <= 7.0
                reverse_val = df_input['true_reverse_gap'].abs()
                reverse_bonus = np.where(is_capable, reverse_val * 1.5, -reverse_val * 0.5)
                promo_penalty = np.where(df_input['is_promotion_first'], -5.0, 0.0)
                dirt_penalty = np.where(df_input['is_first_dirt'], -6.0, 0.0)
                turf_penalty = np.where(df_input['is_first_turf'], -6.0, 0.0)
                score = rank_score + l3f_bonus + reverse_bonus + promo_penalty + dirt_penalty + turf_penalty + df_input['jockey_win_rate'] * 2.0 + (1.0 / np.log1p(df_input['odds'])) * 1.5
        else:
            rank_score = (15.0 - df_input['past_avg_rank']).clip(lower=-10.0) * 4.0
            l3f_bonus = (38.0 - df_input['last_3f']).clip(lower=0) * 2.0
            is_capable = df_input['past_avg_rank'] <= 7.0
            reverse_val = df_input['true_reverse_gap'].abs()
            reverse_bonus = np.where(is_capable, reverse_val * 1.5, -reverse_val * 0.5)
            promo_penalty = np.where(df_input['is_promotion_first'], -5.0, 0.0)
            dirt_penalty = np.where(df_input['is_first_dirt'], -6.0, 0.0)
            turf_penalty = np.where(df_input['is_first_turf'], -6.0, 0.0)
            score = rank_score + l3f_bonus + reverse_bonus + promo_penalty + dirt_penalty + turf_penalty + df_input['jockey_win_rate'] * 2.0 + (1.0 / np.log1p(df_input['odds'])) * 1.5

        exp_s = np.exp(score - score.max())
        df_input['win_prob'] = (exp_s / exp_s.sum()) * 100

        df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
        st.balloons()
        st.subheader("🎯 ガチAI予測結果ランキング（昇級初戦・初ダート・初芝割引反映版）")

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
            t_gap = row.get('true_reverse_gap', 0.0)
            
            is_promo = row.get('is_promotion_first', False)
            prev_c = row.get('prev_class_name', '')
            is_dirt = row.get('is_first_dirt', False)
            is_turf = row.get('is_first_turf', False)

            condition_tag = ""
            if is_promo:
                condition_tag += f" ⚡️【昇級初戦 (前走:{prev_c})】"
            if is_dirt:
                condition_tag += " ⚠️【初ダート】"
            if is_turf:
                condition_tag += " ⚠️【初芝】"

            is_short = float(p_distance) <= 1200 or pd.isna(c_1st)
            corner_str = f"4角: {c_4th:.1f}" if is_short else f"通過順[1角->4角]: {c_1st:.1f}->{c_4th:.1f}"

            st.write(f"**第 {idx+1} 位**: 馬番 {u_num} 🐴 {h_sex}{h_age} **{h_name}**{condition_tag} (予測勝率: **{h_prob:.2f}%** / 調整平均着順: {h_avg:.1f}着 / {corner_str} / **真の逆行度: {t_gap:+.1f}** / 上がり3F: {h_l3f:.1f}秒 / オッズ: {h_odds}倍 / 騎手: {h_jockey})")

with tab2:
    st.subheader("📝 レース結果をPart2マスターに追加する")
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

            new_data_list.append({
                'year': r_year, 'month': r_month, 'day': r_day,
                'place': race_place, 'track': track_type, 'distance': distance, 'condition': condition,
                'race_class': race_class, 'waku': ((u_num-1)//2)+1, 'umaban': u_num, 'name': clean_str(r_name),
                'sex': '牡', 'age': r_age, 'jockey': r_jockey, 'sire': r_sire, 'weight': r_weight,
                'rank': r_rank, 'odds': r_odds, 'popularity': r_pop, 'blinker': r_blinker,
                'corner': r_corner, 'corner_1st': c_1st_val, 'corner_4th': c_4th_val,
                'time': r_time, 'time_sec': parse_time_to_sec(r_time), 'last_3f': r_l3f
            })

    if st.button("🚀 追加データをPart2マスターに保存する！"):
        df_new = pd.DataFrame(new_data_list)
        df_combined = pd.concat([df_m_auto, df_new], ignore_index=True) if not df_m_auto.empty else df_new
        df_combined.to_csv('keiba_master_data_part2.csv', index=False, encoding='cp932')
        st.cache_data.clear()
        st.balloons()
        st.success("🎉 結果データがPart2マスターに保存されました！")

with tab3:
    st.subheader("📝 ガチAIを再学習させる")
    if not df_m_auto.empty:
        st.success(f"📂 Part2マスターデータ読み込み成功！ (読込行数: {len(df_m_auto):,})")
        if st.button("🚀 AIを再学習・アップデートする！"):
            try:
                import lightgbm as lgb
                df_train = df_m_auto.copy().loc[:, ~df_m_auto.columns.duplicated()]
                for col_name, default_val in [('rank', 1), ('last_3f', 35.0), ('distance', 2000.0), ('condition', '良'), ('corner_1st', 5.0), ('corner_4th', 7.0), ('true_reverse_gap', 0.0), ('race_class', '1勝クラス')]:
                    if col_name not in df_train.columns:
                        df_train[col_name] = default_val

                if len(df_train) > 10000: df_train = df_train.sample(n=10000, random_state=42)
                df_train['target'] = (pd.to_numeric(df_train['rank'], errors='coerce') == 1).astype(int)

                for col in ['place', 'track', 'condition', 'sire', 'race_class']:
                    if col in df_train.columns:
                        df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))

                features = ['odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate', 'place', 'track', 'condition', 'sire', 'blinker', 'corner_1st', 'corner_4th', 'true_reverse_gap', 'race_class', 'time_sec', 'last_3f']
                for f in features:
                    if f not in df_train.columns: df_train[f] = 0

                X = df_train[features].fillna(0)
                y = df_train['target']
                clf = lgb.LGBMClassifier(random_state=42)
                clf.fit(X, y)
                joblib.dump(clf, 'keiba_ai_model.pkl')
                st.balloons()
                st.success("🎉 再学習完了!")
            except Exception as e:
                st.warning(f"⚠️ 学習エラー: {e}")
    else:
        st.warning("⚠️ Part2マスターデータが見つかりません。")
