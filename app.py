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

st.title("🐎【完全版・展開ギャップ評価エンジン】競馬AIマスターアプリ")
st.write("マスターデータの列構造完全同期＆レース単位の展開ギャップ算出中！✨🔥")

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
                  elif clean_c in ['上がり3F', '上り3F', 'last_3f', 'L3F', '上がり', '上り', '上がり3Fタイム', '上り3Fタイム']: col_mapping[c] = 'last_3f'
                  elif clean_c in ['距離', 'Distance', 'distance']: col_mapping[c] = 'distance'
                  elif clean_c in ['馬場', '馬場状態', 'Condition', 'condition']: col_mapping[c] = 'condition'
                  elif clean_c in ['性別', 'Sex', 'sex']: col_mapping[c] = 'sex'
                  elif clean_c in ['通過順1', 'コーナー1', '角1']: col_mapping[c] = 'corner_1st'
                  elif clean_c in ['通過順2', 'コーナー2', '角2']: col_mapping[c] = 'corner_2nd'
                  elif clean_c in ['通過順3', 'コーナー3', '角3']: col_mapping[c] = 'corner_3rd'
                  elif clean_c in ['通過順4', 'コーナー4', '角4']: col_mapping[c] = 'corner_4th'
                  elif clean_c in ['年', 'year']: col_mapping[c] = 'year'
                  elif clean_c in ['月', 'month']: col_mapping[c] = 'month'
                  elif clean_c in ['日', 'day']: col_mapping[c] = 'day'
                  elif clean_c in ['場所', '開催', 'place']: col_mapping[c] = 'place'
                  elif clean_c in ['レース番', 'レース番号', 'R', 'race_no']: col_mapping[c] = 'race_no'
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
  else:
      df_m['last_3f'] = np.nan

  if 'distance' in df_m.columns:
      df_m['distance'] = pd.to_numeric(df_m['distance'], errors='coerce').fillna(2000.0)
  else:
      df_m['distance'] = 2000.0

  for c_col in ['corner_1st', 'corner_2nd', 'corner_3rd', 'corner_4th']:
      if c_col in df_m.columns:
          df_m[c_col] = pd.to_numeric(df_m[c_col], errors='coerce')
      else:
          df_m[c_col] = np.nan

  df_m['time_sec'] = df_m['time'].apply(parse_time_to_sec) if 'time' in df_m.columns else 0.0

  # --- レースごとのグループ化キー（年、月、日、場所、レース番）を確実に数値・文字列化 ---
  for k in ['year', 'month', 'day', 'race_no']:
      if k in df_m.columns:
          df_m[k] = pd.to_numeric(df_m[k], errors='coerce').fillna(0).astype(int)
  if 'place' in df_m.columns:
      df_m['place'] = df_m['place'].astype(str).str.strip()

  race_keys = [k for k in ['year', 'month', 'day', 'place', 'race_no'] if k in df_m.columns]

  if len(race_keys) >= 2:
      def calc_race_bias(group):
          top3 = group[group['rank'] <= 3]
          if not top3.empty and 'corner_4th' in top3.columns:
              return top3['corner_4th'].mean()
          return np.nan

      race_bias_series = df_m.groupby(race_keys).apply(calc_race_bias)
      # マルチインデックスの結合エラーを防ぐため map を安全に適用
      df_m['race_top3_c4_avg'] = df_m.set_index(race_keys).index.map(race_bias_series).values
  else:
      df_m['race_top3_c4_avg'] = np.nan

  # 1〜3着の4角平均と自身の4角位置・着順からギャップを計算
  if 'race_top3_c4_avg' in df_m.columns and 'corner_4th' in df_m.columns:
      df_m['race_top3_c4_avg'] = df_m['race_top3_c4_avg'].fillna(5.0)
      # 展開逆行ギャップ: (レースの1~3着平均4角 - 自馬の4角) * (4 - 着順) のイメージ
      df_m['bias_gap'] = (df_m['race_top3_c4_avg'] - df_m['corner_4th']) * (4.0 - df_m['rank'])
  else:
      df_m['bias_gap'] = 0.0

  jockey_win_rates = {}
  if 'jockey' in df_m.columns and 'rank' in df_m.columns:
      j_stats = df_m.groupby('jockey').agg(total=('rank', 'count'), wins=('rank', lambda x: (x == 1).sum()))
      for j, row in j_stats.iterrows():
          if row['total'] > 0: jockey_win_rates[j] = row['wins'] / row['total']

  horse_history_features = {}
  if 'name' in df_m.columns:
      h_grouped = df_m.groupby('name').agg(
          avg_rank=('rank', 'mean'),
          best_rank=('rank', 'min'),
          avg_time=('time_sec', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0.0),
          avg_last_3f=('last_3f', 'mean'),
          race_count=('rank', 'count'),
          sex=('sex', lambda x: x.iloc[0] if len(x) > 0 and pd.notna(x.iloc[0]) else '牡'),
          avg_c1=('corner_1st', 'mean'),
          avg_c2=('corner_2nd', 'mean'),
          avg_c3=('corner_3rd', 'mean'),
          avg_c4=('corner_4th', 'mean'),
          avg_bias_gap=('bias_gap', 'mean')
      )
      for h_name, row in h_grouped.iterrows():
          c_name = clean_str(h_name)
          if c_name:
              horse_history_features[c_name] = {
                  'avg_rank': row['avg_rank'] if not np.isnan(row['avg_rank']) else 5.0,
                  'best_rank': row['best_rank'] if not np.isnan(row['best_rank']) else 5.0,
                  'avg_time': row['avg_time'] if not np.isnan(row['avg_time']) else 0.0,
                  'avg_last_3f': row['avg_last_3f'] if not np.isnan(row['avg_last_3f']) else 35.0,
                  'race_count': row['race_count'],
                  'sex': row['sex'] if str(row['sex']).strip() in ['牡', '牝', 'セン', 'セ'] else '牡',
                  'avg_c1': row['avg_c1'] if not np.isnan(row['avg_c1']) else np.nan,
                  'avg_c2': row['avg_c2'] if not np.isnan(row['avg_c2']) else np.nan,
                  'avg_c3': row['avg_c3'] if not np.isnan(row['avg_c3']) else np.nan,
                  'avg_c4': row['avg_c4'] if not np.isnan(row['avg_c4']) else np.nan,
                  'avg_bias_gap': row['avg_bias_gap'] if not np.isnan(row['avg_bias_gap']) else 0.0
              }

  return df_m, jockey_win_rates, horse_history_features

df_m_auto, jockey_win_rates, horse_history_features = load_and_process_master_data()

tab1, tab2, tab3 = st.tabs(["🚀 ガチ予測", "📝 レース結果を追加", "🧠 AI再学習"])

with tab1:
  st.subheader("🚀 勝ち馬のガチ予測（展開ギャップ完全連動版）")

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
          lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip() != ""]
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
                          if clean_b and not any(kw in clean_b for kw in ["人気", "データベース", "牡", "牝", "セ", "セン", "kg", "斤量", "父", "母", "馬主", "調教師", "生産者", "全成績", "収得賞金"]) and len(clean_b) >= 2 and not clean_b.isdigit():
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
                              right_tokens = parts[1].strip().split()
                              if len(right_tokens) >= 2:
                                  jockey = right_tokens[0]
                                  try: weight = float(right_tokens[1])
                                  except: pass
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
                          try: popularity = int(bl.replace("人気", "").strip())
                          except: pass

                      if "." in bl and not any(kw in bl for kw in ["人気", "kg", "("]) and len(bl) <= 6:
                          try:
                              val = float(bl)
                              if 0.1 <= val < 3000: odds = val
                          except: pass

                  clean_h_name = clean_str(h_name)
                  if clean_h_name not in horse_history_features:
                      for m_name in horse_history_features.keys():
                          if clean_h_name in m_name or m_name in clean_h_name:
                              clean_h_name = m_name
                              break

                  matched_hist = horse_history_features.get(clean_h_name, {
                      'avg_rank': 5.0, 'best_rank': 5.0, 'avg_time': 0.0, 'avg_last_3f': 35.0,
                      'race_count': 0, 'sex': sex, 'avg_c1': np.nan, 'avg_c2': np.nan, 'avg_c3': np.nan, 'avg_c4': np.nan, 'avg_bias_gap': 0.0
                  })

                  if matched_hist.get('sex') in ['牡', '牝', 'セン', 'セ']:
                      sex = matched_hist['sex']

                  c1 = matched_hist['avg_c1'] if not np.isnan(matched_hist['avg_c1']) else float((umaban % 5) + 2)
                  c2 = matched_hist['avg_c2'] if not np.isnan(matched_hist['avg_c2']) else c1
                  c3 = matched_hist['avg_c3'] if not np.isnan(matched_hist['avg_c3']) else c1
                  c4 = matched_hist['avg_c4'] if not np.isnan(matched_hist['avg_c4']) else c1
                  b_gap = matched_hist['avg_bias_gap']

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
                      'corner_1st': c1,
                      'corner_2nd': c2,
                      'corner_3rd': c3,
                      'corner_4th': c4,
                      'bias_gap': b_gap
                  })
                  i = j - 1
              i += 1
  except Exception as e:
      input_data_list = []

  if len(input_data_list) == 0:
      st.warning("⚠️ テキスト未入力または解析対象外のため、デフォルトの8頭で表示しています。")
      for i in range(8):
          input_data_list.append({
              'place': p_place, 'track': p_track, 'distance': p_distance, 'condition': p_condition,
              'race_class': p_class, 'waku': 1, 'umaban': i+1, 'name': f"馬番{i+1}", 'sex': '牡', 'age': 4, 'sire': '不明',
              'odds': float(i+2), 'popularity': i+1, 'weight': 56.0, 'jockey': '不明', 'jockey_win_rate': 0.08,
              'past_avg_rank': 5.0, 'past_best_rank': 5.0, 'time_sec': 0.0, 'last_3f': 35.0, 'blinker': 0,
              'corner_1st': float(i+1), 'corner_2nd': float(i+1), 'corner_3rd': float(i+1), 'corner_4th': float(i+1),
              'bias_gap': 0.0
          })
  else:
      matched_count = sum(1 for x in input_data_list if x['past_avg_rank'] != 5.0)
      st.success(f"✨ テキストから出走馬 **{len(input_data_list)}頭** を検出！(うちマスター一致: **{matched_count}頭**)")

  if st.button("🚀 ガチ予測を実行する！"):
      df_input = pd.DataFrame(input_data_list)
      df_input['odds'] = pd.to_numeric(df_input['odds'], errors='coerce').fillna(10.0)
      df_input['popularity'] = pd.to_numeric(df_input['popularity'], errors='coerce').fillna(99)
      df_input['distance'] = float(p_distance)
      df_input['condition'] = str(p_condition)

      avg_field_c4 = df_input['corner_4th'].mean()
      df_input['pace_bias'] = avg_field_c4 - df_input['corner_4th']

      if model is not None:
          try:
              df_full = pd.concat([df_m_auto, df_input], ignore_index=True) if not df_m_auto.empty else df_input
              df_full = df_full.loc[:, ~df_full.columns.duplicated()]
              for col in ['corner_1st', 'corner_2nd', 'corner_3rd', 'corner_4th', 'pace_bias', 'bias_gap']:
                  if col not in df_full.columns: df_full[col] = 0.0

              for col in ['place', 'track', 'condition', 'sire', 'race_class']:
                  if col in df_full.columns:
                      df_full[col] = LabelEncoder().fit_transform(df_full[col].astype(str))
              df_input_enc = df_full.tail(len(df_input)).copy()

              features = [
                  'odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate',
                  'place', 'track', 'condition', 'sire', 'blinker',
                  'corner_1st', 'corner_2nd', 'corner_3rd', 'corner_4th', 'pace_bias', 'bias_gap',
                  'race_class', 'time_sec', 'last_3f'
              ]
              for f in features:
                  if f not in df_input_enc.columns: df_input_enc[f] = 0
              model_probs = model.predict_proba(df_input_enc[features].fillna(0))[:, 1]

              score = model_probs * 1.5 + (6.0 - df_input['past_avg_rank']).clip(lower=0) * 1.0 + (38.0 - df_input['last_3f']).clip(lower=0) * 1.0 + df_input['bias_gap'].clip(lower=-5, upper=5) * 1.5 + df_input['jockey_win_rate'] * 1.0 + (1.0 / np.log1p(df_input['odds'])) * 0.8
              exp_s = np.exp(score - score.max())
              df_input['win_prob'] = (exp_s / exp_s.sum()) * 100
          except Exception as e:
              score = (6.0 - df_input['past_avg_rank']).clip(lower=0) * 1.0 + (38.0 - df_input['last_3f']).clip(lower=0) * 1.0 + df_input['bias_gap'].clip(lower=-5, upper=5) * 1.5 + df_input['jockey_win_rate'] * 1.0 + (1.0 / np.log1p(df_input['odds'])) * 1.0
              exp_s = np.exp(score - score.max())
              df_input['win_prob'] = (exp_s / exp_s.sum()) * 100
      else:
          score = (6.0 - df_input['past_avg_rank']).clip(lower=0) * 1.0 + (38.0 - df_input['last_3f']).clip(lower=0) * 1.0 + df_input['bias_gap'].clip(lower=-5, upper=5) * 1.5 + df_input['jockey_win_rate'] * 1.0 + (1.0 / np.log1p(df_input['odds'])) * 1.0
          exp_s = np.exp(score - score.max())
          df_input['win_prob'] = (exp_s / exp_s.sum()) * 100

      df_input = df_input.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
      st.balloons()
      st.subheader("🎯 ガチAI予測結果ランキング（展開ギャップ完全連動版）")

      for idx, row in df_input.iterrows():
          u_num = row.get('umaban', idx+1)
          h_sex = row.get('sex', '牡')
          h_age = row.get('age', 4)
          h_name = row.get('name', f'馬番{u_num}')
          h_prob = row.get('win_prob', 0.0)
          h_avg = row.get('past_avg_rank', 5.0)
          h_l3f = row.get('last_3f', 35.0)
          h_bgap = row.get('bias_gap', 0.0)
          h_odds = row.get('odds', 10.0)
          h_jockey = row.get('jockey', '不明')
          c1 = row.get('corner_1st', 8.0)
          c4 = row.get('corner_4th', 8.0)

          st.write(f"**第 {idx+1} 位**: 馬番 {u_num} 🐴 {h_sex}{h_age} **{h_name}** (予測勝率: **{h_prob:.2f}%** / 直近平均着順: {h_avg:.1f}着 / 通過順[1角->4角]: {c1:.1f}→{c4:.1f} / 展開逆行ギャップ: {h_bgap:+.1f} / 上がり3F: {h_l3f:.1f}秒 / オッズ: {h_odds}倍 / 騎手: {h_jockey})")

with tab2:
  st.subheader("📝 レース結果をPart2マスターに追加する")
  col_d1, col_d2, col_d3 = st.columns(3)
  with col_d1: r_year = st.number_input("年", min_value=2000, max_value=2030, value=2024, key="r_year")
  with col_d2: r_month = st.number_input("月", min_value=1, max_value=12, value=7, key="r_month")
  with col_d3: r_day = st.number_input("日", min_value=1, max_value=31, value=20, key="r_day")

  col_r1, col_r2, col_r3 = st.columns(3)
  with col_r1:
      race_place = st.selectbox("開催場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="r_place")
      race_class = st.selectbox("クラス", ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "G3", "G2", "G1"], key="r_class")
  with col_r2:
      track_type = st.selectbox("トラック", ["芝", "ダート", "障害"], key="r_track")
      distance = st.number_input("距離 (m)", value=1200, step=100, key="r_distance")
  with col_r3:
      condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"], key="r_cond")

  res_num_horses = st.slider("出走頭数", min_value=1, max_value=18, value=5, key="res_num")
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
              r_weight = st.number_input("斤量", value=54.0, step=0.5, key=f"r_weight_{i}")
              r_sire = st.text_input("父馬名", "不明", key=f"r_sire_{i}")
              r_age = st.number_input("年齢", min_value=2, max_value=15, value=3, key=f"r_age_{i}")
              c1_in = st.number_input("1角通過順", min_value=1.0, max_value=18.0, value=3.0, step=1.0, key=f"r_c1_{i}")
              c2_in = st.number_input("2角通過順", min_value=1.0, max_value=18.0, value=3.0, step=1.0, key=f"r_c2_{i}")
              c3_in = st.number_input("3角通過順", min_value=1.0, max_value=18.0, value=3.0, step=1.0, key=f"r_c3_{i}")
              c4_in = st.number_input("4角通過順", min_value=1.0, max_value=18.0, value=3.0, step=1.0, key=f"r_c4_{i}")
              r_time = st.text_input("走破タイム (例: 1:09.2)", "1:10.0", key=f"r_time_{i}")
              r_l3f = st.number_input("上がり3Fタイム (例: 34.5)", min_value=25.0, max_value=50.0, value=35.0, step=0.1, key=f"r_l3f_{i}")

          new_data_list.append({
              'year': r_year, 'month': r_month, 'day': r_day,
              'place': race_place, 'track': track_type, 'distance': distance, 'condition': condition,
              'race_class': race_class, 'race_no': 1, 'waku': ((u_num-1)//2)+1, 'umaban': u_num, 'name': clean_str(r_name),
              'sex': '牝', 'age': r_age, 'jockey': r_jockey, 'sire': r_sire, 'weight': r_weight,
              'rank': r_rank, 'odds': r_odds, 'popularity': r_pop, 'blinker': r_blinker,
              'corner_1st': c1_in, 'corner_2nd': c2_in, 'corner_3rd': c3_in, 'corner_4th': c4_in,
              'time': r_time, 'time_sec': parse_time_to_sec(r_time), 'last_3f': r_l3f
          })

  if st.button("🚀 追加データをPart2マスターに保存する！"):
      df_new = pd.DataFrame(new_data_list)
      df_combined = pd.concat([df_m_auto, df_new], ignore_index=True) if not df_m_auto.empty else df_new
      df_combined.to_csv('keiba_master_data_part2.csv', index=False, encoding='cp932')
      st.balloons()
      st.success("🎉 結果がPart2マスターに保存されました！")

with tab3:
  st.subheader("🧠 ガチAIを再学習させる")
  if not df_m_auto.empty:
      st.success(f"📂 Part2マスターデータ読み込み成功！ (読込行数: {len(df_m_auto):,})")
      if st.button("🚀 AIを再学習・アップデートする！"):
          try:
              import lightgbm as lgb
              df_train = df_m_auto.copy().loc[:, ~df_m_auto.columns.duplicated()]
              for col in ['rank', 'last_3f', 'distance', 'corner_1st', 'corner_2nd', 'corner_3rd', 'corner_4th', 'bias_gap']:
                  if col not in df_train.columns: df_train[col] = 0.0
              if len(df_train) > 10000: df_train = df_train.sample(n=10000, random_state=42)
              df_train['target'] = (pd.to_numeric(df_train['rank'], errors='coerce') == 1).astype(int)

              for col in ['place', 'track', 'condition', 'sire', 'race_class']:
                  if col in df_train.columns:
                      df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))

              features = [
                  'odds', 'popularity', 'weight', 'age', 'waku', 'umaban', 'distance', 'jockey_win_rate',
                  'place', 'track', 'condition', 'sire', 'blinker',
                  'corner_1st', 'corner_2nd', 'corner_3rd', 'corner_4th', 'pace_bias', 'bias_gap',
                  'race_class', 'time_sec', 'last_3f'
              ]
              for f in features:
                  if f not in df_train.columns: df_train[f] = 0

              X = df_train[features].fillna(0)
              y = df_train['target']
              clf = lgb.LGBMClassifier(random_state=42)
              clf.fit(X, y)
              joblib.dump(clf, 'keiba_ai_model.pkl')
              st.balloons()
              st.success("🎉 再学習完了！")
          except Exception as e:
              st.warning(f"⚠️ 学習エラー: {e}")
  else:
      st.warning("⚠️ Part2マスターデータが見つかりません。")
