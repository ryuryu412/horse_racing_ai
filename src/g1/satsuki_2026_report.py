"""
皐月賞 2026 全馬詳細診断レポート
docs/g1/satsuki_2026.html を生成
"""
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent

# -----------------------------------------------------------------------
# 過去13年の皐月賞データから導出したスコアテーブル
# -----------------------------------------------------------------------
PREP_WIN_RATE = {
    (9, 1): 33.3, (9, 2): 0.0,  (9, 3): 0.0,
    (8, 1):  4.0, (8, 2): 4.5,  (8, 3): 0.0,
    (7, 1): 16.2, (7, 2): 16.7, (7, 3): 0.0,
    (6, 1):  0.0, (6, 2): 0.0,
}
INTERVAL_RATE = [(9, 13, 13.9), (13, 999, 11.5), (7, 9, 6.2), (0, 5, 3.2), (5, 7, 1.8)]
TI_PEAK_RATE  = [(68, 999, 20.0), (63, 68, 3.8), (58, 63, 8.0), (0, 58, 3.7)]
UR_PEAK_RATE  = [(65, 999, 2.8), (60, 65, 8.3), (55, 60, 5.6), (0, 55, 2.9)]
STYLE_RATE    = {1: 9.5, 2: 4.6, 3: 3.2, 4: 0.0}
STYLE_LABEL   = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}
SIRE_RATE = {
    "キタサンブラック": 25.0, "ドレフォン": 33.3, "オルフェーヴル": 25.0,
    "フジキセキ": 25.0, "エピファネイア": 14.3, "ロードカナロア": 14.3,
    "キズナ": 11.1, "ディープインパクト": 8.8, "キングカメハメハ": 6.2,
}
WEIGHT_RATE = [(5, 999, 4.5), (-4, 5, 7.6), (-999, -4, 3.4)]

WEIGHTS = {
    "model":    0.35,  # 普段モデル（距離/クラスランカー）を軸
    "ti":       0.30,  # タイム指数ピーク（rho=0.23、グリッドサーチ最適）
    "prep":     0.19,  # 前走クラス×着順（rho=0.21）
    "jockey":   0.10,  # 騎手勝率（rho=0.49だが逆因果割引・v3で縮小）
    "interval": 0.03,  # 間隔・成長補正（参考程度）
    "ur":       0.03,  # 上り3F指数（参考程度）
    # style/waku/weight/sire は相関低・サンプル不足のため削除
}
NORM = {
    "model": 1.0, "jockey": 0.25,
    "prep": 33.3, "ti": 20.0, "ur": 8.3,
    "interval": 13.9,
}

CLS_LABEL = {9: "G1", 8: "G2", 7: "G3", 6: "OP", 5: "3勝", 4: "2勝", 3: "1勝"}


def lookup(val, table):
    for lo, hi, r in table:
        if lo <= val < hi:
            return r
    return table[-1][2]


def score_horse(h):
    s, d = {}, {}

    cls = int(float(h.get("1走前_クラス_rank", 0) or 0))
    pos = int(float(h.get("1走前_着順_num", 99) or 99))
    prep = PREP_WIN_RATE.get((cls, min(pos, 6)), PREP_WIN_RATE.get((cls, 1), 2.0))
    s["prep"] = min(prep / NORM["prep"], 1.0)
    d["prep"] = {"rate": prep, "label": f"{CLS_LABEL.get(cls,'-')} {pos}着", "max": 33.3}

    ti1 = float(h.get("1走前_タイム指数", 0) or 0)
    ti2 = float(h.get("2走前_タイム指数", 0) or 0)
    ti_pk = max(ti1, ti2)
    ti_r = lookup(ti_pk, TI_PEAK_RATE)
    s["ti"] = ti_r / NORM["ti"]
    d["ti"] = {"rate": ti_r, "label": f"ピーク {ti_pk:.1f}", "max": 20.0,
               "sub": f"1走前:{ti1:.1f} / 2走前:{ti2:.1f}"}

    ur1 = float(h.get("1走前_上り3F_指数", 0) or 0)
    ur2 = float(h.get("2走前_上り3F_指数", 0) or 0)
    ur_pk = max(ur1, ur2)
    ur_r = lookup(ur_pk, UR_PEAK_RATE)
    s["ur"] = ur_r / NORM["ur"]
    d["ur"] = {"rate": ur_r, "label": f"ピーク {ur_pk:.1f}", "max": 8.3,
               "sub": f"1走前:{ur1:.1f} / 2走前:{ur2:.1f}", "warn": ur_pk >= 65}

    iv = float(h.get("間隔", 0) or 0)
    iv_r = lookup(iv, INTERVAL_RATE)
    if iv >= 13:
        iv_comment = "冬明け・成長余地大"
    elif iv >= 9:
        iv_comment = "適度な間隔（黄金ゾーン）"
    elif iv >= 7:
        iv_comment = "中間隔"
    elif iv >= 5:
        iv_comment = "短め（スプリングS組）"
    else:
        iv_comment = "短間隔・連闘気味"
    s["interval"] = iv_r / NORM["interval"]
    d["interval"] = {"rate": iv_r, "label": f"{int(iv)}週 / {iv_comment}", "max": 13.9}

    jk_wr = float(h.get("騎手_勝率", 0) or 0)
    s["jockey"] = min(jk_wr / NORM["jockey"], 1.0)
    jk_name = str(h.get("騎手", "-") or "-")
    d["jockey"] = {"rate": jk_wr * 100, "label": f"{jk_name}（全体勝率{jk_wr*100:.1f}%）", "max": 25.0}

    cur_r = int(h.get("cur_ランカー順位", 18) or 18)
    sub_r = int(h.get("sub_ランカー順位", 18) or 18)
    cur_d = float(h.get("cur_偏差値の差", 0) or 0)
    sub_d = float(h.get("sub_偏差値の差", 0) or 0)
    mdl = ((19 - cur_r) + (19 - sub_r)) / (2 * 18)
    s["model"] = mdl
    d["model"] = {"rate": mdl * 100, "label": f"距離{cur_r}位 / クラス{sub_r}位",
                  "max": 100, "sub": f"距離diff:{cur_d:+.1f} / クラスdiff:{sub_d:+.1f}"}

    # 未確定指標(None)を除いて正規化
    total = sum(s[k] * WEIGHTS[k] for k in WEIGHTS if s[k] is not None) \
          / sum(WEIGHTS[k] for k in WEIGHTS if s[k] is not None) * 100
    return total, s, d


def grade_color(pct):
    if pct >= 75: return "#e74c3c", "S"
    if pct >= 60: return "#e67e22", "A"
    if pct >= 45: return "#f1c40f", "B"
    if pct >= 30: return "#2ecc71", "C"
    return "#555", "D"


def bar_html(score_01, color, height=8):
    w = int(min(score_01 * 100, 100))
    return (f'<div style="background:#21262d;border-radius:4px;height:{height}px;margin-top:3px">'
            f'<div style="background:{color};height:{height}px;border-radius:4px;width:{w}%"></div></div>')


def dim_block(key, label, data, score_01):
    pending = data.get("pending", False)
    if pending:
        return f"""
    <div style="background:#1c2128;border-radius:6px;padding:10px 12px;margin-bottom:6px;border:1px dashed #333">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="color:#8b949e;font-size:11px;font-weight:bold">{label}</span>
        <span style="color:#555;font-size:12px">未確定</span>
      </div>
      <div style="color:#555;font-size:12px;margin-top:4px">{data['label']}</div>
      <div style="color:#444;font-size:11px">スコア計算から除外（当日確定後に反映）</div>
    </div>"""
    color, _ = grade_color(score_01 * 100)
    pct = score_01 * 100
    warn = data.get("warn", False)
    warn_icon = ' <span style="color:#e74c3c;font-size:10px">⚠ 高すぎ</span>' if warn else ""
    notable = data.get("notable", False)
    notable_icon = ' <span style="color:#f0a500;font-size:10px">★得意血統</span>' if notable else ""
    sub = data.get("sub", "")
    sub_html = f'<div style="color:#555;font-size:10px;margin-top:2px">{sub}</div>' if sub else ""
    rate_str = f"{data['rate']:.1f}%" if data['rate'] is not None else "-"
    return f"""
    <div style="background:#1c2128;border-radius:6px;padding:10px 12px;margin-bottom:6px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="color:#8b949e;font-size:11px;font-weight:bold">{label}</span>
        <span style="color:{color};font-weight:bold;font-size:13px">{pct:.0f}点</span>
      </div>
      <div style="color:#c9d1d9;font-size:12px;margin-top:4px">{data['label']}{warn_icon}{notable_icon}</div>
      <div style="color:{color};font-size:11px">勝率参考値: {rate_str} (基準:{data['max']:.1f}%)</div>
      {sub_html}
      {bar_html(score_01, color)}
    </div>"""


def horse_card(rank, h_data, total, scores, details, odds):
    color, grade = grade_color(total)
    odds_str = f"{odds:.1f}" if isinstance(odds, float) and not np.isnan(odds) else "-"
    pop = h_data.get("人気", "-")
    pop_str = f"{int(pop)}番人気" if not pd.isna(pop) else "-"
    jockey = h_data.get("騎手", "-")
    sire = str(h_data.get("種牡馬", "") or "")
    dam_sire = str(h_data.get("母父名", "") or "")

    dim_labels = {
        "model":    "AIモデル（距離/クラス）",
        "jockey":   "騎手（全体勝率）",
        "prep":     "前走クラス×着順",
        "ti":       "タイム指数ピーク",
        "interval": "間隔・成長補正",
        "ur":       "上り3F指数",
    }
    dims_html = "".join(dim_block(k, dim_labels[k], details[k], scores[k]) for k in dim_labels if k in details)

    # 総合評価コメント
    strengths = [dim_labels[k] for k in scores if scores[k] is not None and scores[k] >= 0.7]
    weaknesses = [dim_labels[k] for k in scores if scores[k] is not None and scores[k] <= 0.25]
    verdict = ""
    if total >= 75:
        verdict = "複数の重要指標が高水準で揃った有力候補。"
    elif total >= 55:
        verdict = "いくつかの指標に強みがあるが、弱点も存在する中上位評価。"
    elif total >= 40:
        verdict = "指標にムラがあり一発狙いの穴馬候補。"
    else:
        verdict = "多くの指標が低水準。過去データ的には苦しい条件が重なっている。"
    if strengths:
        verdict += f" 強み：{' / '.join(strengths[:3])}。"
    if weaknesses:
        verdict += f" 弱点：{' / '.join(weaknesses[:2])}。"

    rank_badge = ""
    if rank == 1:
        rank_badge = f'<div style="background:#e74c3c;color:white;font-size:20px;font-weight:bold;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center">{rank}</div>'
    elif rank <= 3:
        rank_badge = f'<div style="background:#e67e22;color:white;font-size:18px;font-weight:bold;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center">{rank}</div>'
    else:
        rank_badge = f'<div style="background:#21262d;color:#8b949e;font-size:16px;font-weight:bold;border-radius:50%;width:36px;height:36px;display:flex;align-items:center;justify-content:center">{rank}</div>'

    return f"""
  <div style="background:#161b22;border-radius:10px;padding:20px;margin-bottom:16px;border-left:4px solid {color}">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap">
      {rank_badge}
      <div>
        <div style="font-size:20px;font-weight:bold;color:#fff">{h_data['馬名']}</div>
        <div style="color:#8b949e;font-size:12px;margin-top:2px">
          {jockey} ／ {pop_str} ／ 単勝 {odds_str}倍 ／ {sire}×{dam_sire}
        </div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:32px;font-weight:bold;color:{color}">{total:.1f}</div>
        <div style="background:{color};color:white;border-radius:4px;padding:2px 8px;font-size:13px;font-weight:bold;text-align:center">Grade {grade}</div>
      </div>
    </div>
    {bar_html(total/100, color, 10)}
    <div style="color:#8b949e;font-size:12px;margin:10px 0 12px;line-height:1.7">{verdict}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:6px">
      {dims_html}
    </div>
  </div>"""


WEIGHTS_NO_MODEL = {k: v for k, v in WEIGHTS.items() if k != "model"}
PARQUET_STYLE_MAP = {0: 1, 1: 2, 2: 3, 3: 4}  # 互換性のため残す（未使用）

# 枠番ルックアップ（parquetには馬番列がある想定）
PARQUET_STYLE_MAP = {0: 1, 1: 2, 2: 3, 3: 4}  # parquet encoding → STYLE_RATE key


def score_horse_historical(h) -> float:
    """過去データ用スコアリング（model次元除外 / 枠番はparquet馬番から算出）"""
    s = {}

    def safe_int(val, default):
        try:
            v = float(val)
            return default if np.isnan(v) else int(v)
        except (TypeError, ValueError):
            return default

    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return default if np.isnan(v) else v
        except (TypeError, ValueError):
            return default

    cls = safe_int(h.get("1走前_クラス_rank", 0), 0)
    pos = safe_int(h.get("1走前_着順_num", 99), 99)
    prep = PREP_WIN_RATE.get((cls, min(pos, 6)), PREP_WIN_RATE.get((cls, 1), 2.0))
    s["prep"] = min(prep / NORM["prep"], 1.0)

    ti1 = safe_float(h.get("1走前_タイム指数", 0))
    ti2 = safe_float(h.get("2走前_タイム指数", 0))
    ti_r = lookup(max(ti1, ti2), TI_PEAK_RATE)
    s["ti"] = ti_r / NORM["ti"]

    ur1 = safe_float(h.get("1走前_上り3F_指数", 0))
    ur2 = safe_float(h.get("2走前_上り3F_指数", 0))
    ur_r = lookup(max(ur1, ur2), UR_PEAK_RATE)
    s["ur"] = ur_r / NORM["ur"]

    iv = safe_float(h.get("間隔", 0))
    s["interval"] = lookup(iv, INTERVAL_RATE) / NORM["interval"]

    jk_wr = safe_float(h.get("騎手_勝率", 0))
    s["jockey"] = min(jk_wr / NORM["jockey"], 1.0)

    active = {k: v for k, v in WEIGHTS_NO_MODEL.items() if k in s}
    total = sum(s[k] * active[k] for k in active) / sum(active.values()) * 100
    return total


def compute_elimination_threshold() -> dict:
    """過去皐月賞の全馬スコアリング → 3着以内最低スコアをしきい値として返す"""
    parquet_path = ROOT / "data/processed/all_venues_features.parquet"
    df = pd.read_parquet(parquet_path)
    hist = df[df["レース名"] == "皐月賞G1"].copy()
    hist = hist.dropna(subset=["着順_num"])
    hist["着順_num"] = hist["着順_num"].astype(int)

    scores_all = []
    for _, h in hist.iterrows():
        sc = score_horse_historical(h)
        top3 = int(h["着順_num"]) <= 3
        scores_all.append({
            "馬名": h.get("馬名", ""),
            "日付": str(h.get("日付", "")),
            "着順": int(h["着順_num"]),
            "score": sc,
            "top3": top3,
        })

    df_sc = pd.DataFrame(scores_all)
    top3_df = df_sc[df_sc["top3"]]
    threshold = float(top3_df["score"].min())

    # スコア帯別集計（10点刻み）
    bins = list(range(0, 101, 10))
    labels = [f"{lo}〜{lo+10}" for lo in range(0, 100, 10)]
    df_sc["bucket"] = pd.cut(df_sc["score"], bins=bins, labels=labels, right=False)
    bucket_stats = []
    for label in labels:
        sub = df_sc[df_sc["bucket"] == label]
        total_b = len(sub)
        top3_b = int(sub["top3"].sum())
        rate = top3_b / total_b * 100 if total_b > 0 else 0.0
        bucket_stats.append({"range": label, "total": total_b, "top3": top3_b, "rate": rate})

    # 年別3着以内最低スコア
    df_sc["year"] = df_sc["日付"].str[:4]
    year_min = df_sc[df_sc["top3"]].groupby("year")["score"].min().reset_index()

    return {
        "threshold": threshold,
        "top3_min": threshold,
        "top3_max": float(top3_df["score"].max()),
        "top3_mean": float(top3_df["score"].mean()),
        "all_df": df_sc,
        "bucket_stats": bucket_stats,
        "year_min": year_min,
        "total_horses": len(df_sc),
        "total_top3": len(top3_df),
    }


def elimination_section_html(thresh_data: dict, today_horses: list) -> str:
    # 今年の平均・各馬の平均差を計算
    scores_nm = [item["total_no_model"] for item in today_horses]
    race_mean = float(np.mean(scores_nm))

    def gap_label(gap):
        """平均との差を日本語テキストで返す"""
        if gap >= 0:
            return f"平均より{gap:.1f}点高い", "#2ecc71"
        else:
            return f"平均より{abs(gap):.1f}点低い", "#e74c3c" if gap < -5 else "#f1c40f"

    def zone(gap):
        """平均差に基づく消し判定"""
        if gap < -17:  return "full_elim", "#e74c3c", "3着以内ゼロ圏（過去13年で3着以内なし）"
        if gap < -5:   return "win_elim",  "#e67e22", "1着ゼロ圏（過去13年で1着なし）"
        return                 "ok",        "#8b949e", ""

    # 今年18頭を低い順に
    today_sorted = sorted(today_horses, key=lambda x: x["total_no_model"])

    # ---- 消し馬ブロック ----
    full_elim, win_elim = [], []
    for item in today_sorted:
        gap = item["total_no_model"] - race_mean
        zkey, zcolor, zlabel = zone(gap)
        gl, _ = gap_label(gap)
        if zkey == "full_elim":
            full_elim.append((item["h"]["馬名"], gap, gl))
        elif zkey == "win_elim":
            win_elim.append((item["h"]["馬名"], gap, gl))

    def elim_card(name, gap, gl, color, badge):
        return (f'<div style="background:#1a1a2e;border:1px solid {color};border-radius:8px;'
                f'padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px">'
                f'<span style="background:{color};color:white;border-radius:4px;padding:2px 8px;'
                f'font-size:11px;font-weight:bold;white-space:nowrap">{badge}</span>'
                f'<span style="color:#fff;font-weight:bold;font-size:15px">{name}</span>'
                f'<span style="color:{color};font-size:13px;margin-left:auto">{gl}</span>'
                f'</div>')

    elim_html = ""
    if full_elim:
        elim_html += '<div style="color:#e74c3c;font-size:12px;font-weight:bold;margin-bottom:6px">▼ 3着以内ゼロ圏（過去13年で平均より17点以上低い馬は3着以内なし）</div>'
        elim_html += "".join(elim_card(n, g, gl, "#e74c3c", "完全消し") for n, g, gl in full_elim)
    if win_elim:
        elim_html += '<div style="color:#e67e22;font-size:12px;font-weight:bold;margin:12px 0 6px">▼ 1着ゼロ圏（過去13年で平均より5点以上低い馬は1着なし）</div>'
        elim_html += "".join(elim_card(n, g, gl, "#e67e22", "1着消し") for n, g, gl in win_elim)
    if not full_elim and not win_elim:
        elim_html = '<div style="color:#8b949e;padding:10px 0">消し馬なし</div>'

    # ---- 平均差テーブル ----
    gap_data = [
        ("平均より17点以上低い", None, -17, "3着以内ゼロ",  "0%",    "0%",    "0%"),
        ("平均より5〜17点低い",  -17,   -5, "1着ゼロ",     "〜10%", "〜7%",  "0%"),
        ("平均±5点以内",         -5,    5,  "互角",        "13%",   "8%",   "2.5%"),
        ("平均より5〜10点高い",    5,   10,  "有望",        "34%",   "23%",  "17%"),
        ("平均より10〜15点高い",  10,   15,  "上位",        "41%",   "29%",  "24%"),
        ("平均より15〜20点高い",  15,   20,  "強力",        "55%",   "36%",  "27%"),
        ("平均より20点以上高い",  20,  None, "別格",        "64%",   "55%",  "36%"),
    ]

    gap_rows = ""
    for label, lo, hi, tag, r3, r2, r1 in gap_data:
        today_in = []
        for item in today_horses:
            gap = item["total_no_model"] - race_mean
            in_band = (lo is None or gap >= lo) and (hi is None or gap < hi)
            if in_band:
                today_in.append(item["h"]["馬名"])
        names_html = " / ".join(f'<span style="color:#f0a500;font-weight:bold">{n}</span>' for n in today_in) if today_in else '<span style="color:#444">—</span>'
        r3_color = "#e74c3c" if r3 in ("0%","〜10%") else "#f1c40f" if "%" in r3 and float(r3.replace("〜","").replace("%","")) < 20 else "#2ecc71"
        gap_rows += f"""
      <tr>
        <td style="font-size:12px;color:#c9d1d9">{label}</td>
        <td style="text-align:center;color:#8b949e;font-size:11px">{tag}</td>
        <td style="text-align:center;color:{r3_color};font-weight:bold">{r3}</td>
        <td style="text-align:center;color:{r3_color}">{r2}</td>
        <td style="text-align:center;color:{r3_color}">{r1}</td>
        <td style="font-size:12px">{names_html}</td>
      </tr>"""

    return f"""
<div style="background:#161b22;border-radius:10px;padding:20px;margin-bottom:24px;border-left:4px solid #e74c3c">
  <h2 style="color:#e74c3c;border:none;margin-top:0;padding-left:0">🚫 消し馬分析（平均差方式）</h2>
  <p style="color:#8b949e;font-size:12px;margin-bottom:16px;line-height:1.8">
    過去13年（2013〜2025）の皐月賞全{thresh_data['total_horses']}頭を分析。<br>
    各レースの出走馬平均スコアからの差（平均より何点高い／低い）で判定。<br>
    今年18頭の平均スコア = <strong style="color:#f0a500">{race_mean:.1f}点</strong>
  </p>

  {elim_html}

  <h3 style="color:#8b949e;font-size:13px;margin:20px 0 10px">平均差と着順確率（過去13年の実績）</h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr>
      <th style="background:#21262d;padding:8px;text-align:left">平均との差</th>
      <th style="background:#21262d;padding:8px;text-align:center">判定</th>
      <th style="background:#21262d;padding:8px;text-align:center">3着以内率</th>
      <th style="background:#21262d;padding:8px;text-align:center">2着以内率</th>
      <th style="background:#21262d;padding:8px;text-align:center">1着率</th>
      <th style="background:#21262d;padding:8px;text-align:left">今年の該当馬</th>
    </tr></thead>
    <tbody>{gap_rows}</tbody>
  </table>
  <div style="color:#555;font-size:11px;margin-top:10px">
    ※ スコアは前走データ7指標（AIモデル除外）。サマリー表の総合スコアとは異なります。
  </div>
</div>"""


def compute_jockey_stats() -> dict:
    """過去13年皐月賞の騎手別成績を返す"""
    parquet_path = ROOT / "data/processed/all_venues_features.parquet"
    df = pd.read_parquet(parquet_path)
    hist = df[df["レース名"] == "皐月賞G1"].dropna(subset=["着順_num"]).copy()
    hist["着順_num"] = hist["着順_num"].astype(int)
    grp = hist.groupby("騎手").agg(
        rides=("着順_num", "count"),
        wins=("着順_num", lambda x: (x == 1).sum()),
        top3=("着順_num", lambda x: (x <= 3).sum()),
    ).reset_index()
    grp["win_rate"] = grp["wins"] / grp["rides"] * 100
    grp["top3_rate"] = grp["top3"] / grp["rides"] * 100
    return {row["騎手"]: row.to_dict() for _, row in grp.iterrows()}


def jockey_section_html(jk_stats: dict, today_horses: list) -> str:
    rows = []
    for item in today_horses:
        h = item["h"]
        name = h["馬名"]
        jk = str(h.get("騎手", "-") or "-")
        st = jk_stats.get(jk, {})
        rides = int(st.get("rides", 0))
        wins  = int(st.get("wins", 0))
        top3  = int(st.get("top3", 0))
        t3r   = st.get("top3_rate", None)
        wr    = st.get("win_rate", None)
        # 全体勝率
        jk_overall = float(h.get("騎手_勝率", 0) or 0)

        if rides == 0:
            hist_str = "初騎乗"
            t3_color = "#555"
            bar_w = 0
        else:
            hist_str = f"{top3}/{rides}回 ({t3r:.0f}%)"
            t3_color = "#e74c3c" if t3r >= 40 else "#f1c40f" if t3r >= 20 else "#8b949e"
            bar_w = int(min(t3r * 2, 100))

        overall_bar = int(min(jk_overall * 400, 100))
        overall_color = "#3498db" if jk_overall >= 0.20 else "#8b949e" if jk_overall >= 0.12 else "#555"

        rows.append((item["total"], f"""
      <tr>
        <td style="font-weight:bold;color:#c9d1d9">{name}</td>
        <td style="color:#f0a500;font-weight:bold">{jk}</td>
        <td style="text-align:center;color:{t3_color};font-weight:bold">{hist_str}</td>
        <td style="min-width:100px">
          <div style="background:#21262d;border-radius:3px;height:8px">
            <div style="background:{t3_color};height:8px;border-radius:3px;width:{bar_w}%"></div>
          </div>
        </td>
        <td>
          <div style="display:flex;align-items:center;gap:6px">
            <div style="background:#21262d;border-radius:3px;height:8px;flex:1">
              <div style="background:{overall_color};height:8px;border-radius:3px;width:{overall_bar}%"></div>
            </div>
            <span style="color:{overall_color};font-size:12px;min-width:40px">{jk_overall*100:.1f}%</span>
          </div>
        </td>
      </tr>"""))

    rows.sort(key=lambda x: -x[0])
    rows_html = "".join(r for _, r in rows)

    return f"""
<div style="background:#161b22;border-radius:10px;padding:20px;margin-bottom:24px;border-left:4px solid #f0a500">
  <h2 style="color:#f0a500;border:none;margin-top:0;padding-left:0">🏇 騎手評価</h2>
  <p style="color:#8b949e;font-size:12px;margin-bottom:16px;line-height:1.8">
    皐月賞3着以内率は過去13年の実績。全体勝率（騎手_勝率）は直近の全レースベース。<br>
    分析上、全体勝率は着順との相関が最も高い指標（rho=0.49）でした。
  </p>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr>
      <th style="background:#21262d;padding:8px;text-align:left">馬名</th>
      <th style="background:#21262d;padding:8px;text-align:left">騎手</th>
      <th style="background:#21262d;padding:8px;text-align:center">皐月賞3着率（過去13年）</th>
      <th style="background:#21262d;padding:8px">バー</th>
      <th style="background:#21262d;padding:8px">全体勝率</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div style="color:#555;font-size:11px;margin-top:10px">
    ※ 表は総合スコア順。騎手成績はスコアに未加算（参考情報）。
  </div>
</div>"""


def and_analysis_section_html(today_horses: list) -> str:
    """AND指標分析（高スコア指標の数 → 過去3着率）"""
    EXPECTED = {0: ("11.8%", "1着率2%", "#555"),
                1: ("14.6%", "1着率3%", "#8b949e"),
                2: ("22.2%", "1着率11%", "#f1c40f"),
                3: ("46.7%", "1着率27%", "#27ae60"),
                4: ("参考値以上", "1着率67%超", "#e74c3c")}

    rows_html = ""
    for item in sorted(today_horses, key=lambda x: (
        -sum(1 for k in ["prep","ti","interval","ur"] if (x["scores"].get(k) or 0) >= 0.5),
        -x["total"]
    )):
        h = item["h"]
        sc = item["scores"]
        n_high = sum(1 for k in ["prep","ti","interval","ur"] if (sc.get(k) or 0) >= 0.5)
        exp3, exp1, color = EXPECTED.get(n_high, ("?", "?", "#555"))

        def cell(k):
            v = sc.get(k) or 0
            c = "#2ecc71" if v >= 0.5 else "#555"
            return f'<td style="text-align:center;color:{c};font-weight:bold">{v*100:.0f}</td>'

        rows_html += f"""
      <tr>
        <td style="font-weight:bold;color:{color}">{h["馬名"]}</td>
        <td style="text-align:center;color:{color};font-weight:bold;font-size:16px">{n_high}</td>
        {cell("prep")}{cell("ti")}{cell("interval")}{cell("ur")}
        <td style="color:{color};font-size:12px">{exp3}</td>
        <td style="color:{color};font-size:12px">{exp1}</td>
      </tr>"""

    return f"""
<div style="background:#161b22;border-radius:10px;padding:20px;margin-bottom:24px;border-left:4px solid #27ae60">
  <h2 style="color:#27ae60;border:none;margin-top:0;padding-left:0">📐 AND指標分析（指標の重なり）</h2>
  <p style="color:#8b949e;font-size:12px;margin-bottom:16px;line-height:1.8">
    prep（前走クラス）・ti（タイム指数）・interval（間隔）・ur（上り3F）の4指標で50点超えを「高評価」とし、<br>
    何指標同時に高いかで3着以内率が大きく変わる。加重平均では見えない<strong style="color:#c9d1d9">AND的な強さ</strong>を示す。
  </p>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:20px;text-align:center">
    {"".join(f'<div style="background:#1c2128;border-radius:6px;padding:10px;border-top:2px solid {c}"><div style="color:{c};font-size:18px;font-weight:bold">{n}指標</div><div style="color:#c9d1d9;font-size:13px;margin-top:4px">{e3}</div><div style="color:#8b949e;font-size:11px">{e1}</div></div>' for n,(e3,e1,c) in EXPECTED.items())}
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:13px">
    <thead><tr>
      <th style="background:#21262d;padding:8px;text-align:left">馬名</th>
      <th style="background:#21262d;padding:8px;text-align:center">高指標数</th>
      <th style="background:#21262d;padding:8px;text-align:center">前走<br>クラス</th>
      <th style="background:#21262d;padding:8px;text-align:center">TI<br>ピーク</th>
      <th style="background:#21262d;padding:8px;text-align:center">間隔</th>
      <th style="background:#21262d;padding:8px;text-align:center">上り3F</th>
      <th style="background:#21262d;padding:8px;text-align:center">期待3着率</th>
      <th style="background:#21262d;padding:8px;text-align:center">期待1着率</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div style="color:#555;font-size:11px;margin-top:10px">
    ※ 数値は各指標の0〜100スコア。緑=50超（高評価）。過去実績は2013〜2025年の{227}頭ベース。
  </div>
</div>"""


def generate():
    print("データ読み込み中...")
    with open(ROOT / "data/raw/cache/出馬表形式4月19日オッズcsv.cache.pkl", "rb") as f:
        cache = pickle.load(f)
    result = cache["result"]
    r11 = result[(result["会場"] == "中") & (result["Ｒ"] == 11)].copy()

    print("過去皐月賞スコアリング（消し馬しきい値算出 + 騎手成績）...")
    thresh_data = compute_elimination_threshold()
    jk_stats = compute_jockey_stats()
    print(f"  消しラインしきい値: {thresh_data['threshold']:.1f}点 (3着以内最低)")

    horses = []
    for _, h in r11.iterrows():
        total, sc, det = score_horse(h)
        total_no_model = score_horse_historical(h)
        odds = h.get("単勝オッズ", np.nan)
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            odds = np.nan
        horses.append({"h": h, "total": total, "scores": sc, "details": det,
                       "odds": odds, "total_no_model": total_no_model})

    horses.sort(key=lambda x: -x["total"])

    # 新セクション生成
    jk_html = jockey_section_html(jk_stats, horses)
    and_html = and_analysis_section_html(horses)

    # サマリーテーブル
    summary_rows = ""
    for i, item in enumerate(horses, 1):
        h = item["h"]
        color, grade = grade_color(item["total"])
        odds_str = f"{item['odds']:.1f}" if not np.isnan(item["odds"]) else "-"
        pop = h.get("人気", "-")
        pop_str = str(int(pop)) if not pd.isna(pop) else "-"
        sc = item["scores"]
        summary_rows += f"""
        <tr>
          <td style="text-align:center;font-weight:bold;color:{color}">{i}</td>
          <td><a href="#{h['馬名']}" style="color:#c9d1d9;text-decoration:none;font-weight:bold">{h['馬名']}</a></td>
          <td style="color:#aaa">{h.get('騎手','-')}</td>
          <td style="text-align:center;color:#aaa">{pop_str}</td>
          <td style="color:#f0a500;text-align:center">{odds_str}</td>
          <td style="text-align:center;font-weight:bold;color:{color}">{item['total']:.1f}</td>
          <td style="text-align:center">{'🟢' if sc['model']>=0.7 else '🟡' if sc['model']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['jockey']>=0.7 else '🟡' if sc['jockey']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['prep']>=0.7 else '🟡' if sc['prep']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['ti']>=0.7 else '🟡' if sc['ti']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['interval']>=0.7 else '🟡' if sc['interval']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['ur']>=0.7 else '🟡' if sc['ur']>=0.4 else '🔴'}</td>
        </tr>"""

    # 消し馬分析セクション
    elim_html_section = elimination_section_html(thresh_data, horses)

    # 馬別詳細カード（全馬）
    cards_html = ""
    for i, item in enumerate(horses, 1):
        cards_html += f'<div id="{item["h"]["馬名"]}">' + horse_card(i, item["h"], item["total"], item["scores"], item["details"], item["odds"]) + "</div>"

    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>皐月賞 2026 全馬詳細診断レポート</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Noto Sans JP',sans-serif; margin:0; padding:20px; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  h1 {{ color:#f0a500; border-bottom:2px solid #f0a500; padding-bottom:10px; font-size:22px; }}
  h2 {{ color:#8b949e; font-size:15px; border-left:3px solid #f0a500; padding-left:8px; margin:24px 0 12px; }}
  .meta {{ color:#555; font-size:12px; margin-bottom:24px; line-height:1.8; }}
  .summary-table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; font-size:13px; margin-bottom:32px; }}
  .summary-table th {{ background:#21262d; color:#8b949e; padding:8px 6px; font-size:11px; text-align:center; }}
  .summary-table td {{ padding:8px 6px; border-bottom:1px solid #21262d; }}
  .summary-table tr:hover {{ background:#1c2128; }}
  .findings {{ background:#161b22; border-radius:8px; padding:16px 20px; margin-bottom:24px; }}
  .findings li {{ line-height:2.2; font-size:13px; }}
  .weight-table {{ width:100%; border-collapse:collapse; font-size:12px; margin-bottom:24px; }}
  .weight-table th {{ background:#21262d; color:#8b949e; padding:6px 10px; text-align:left; }}
  .weight-table td {{ padding:6px 10px; border-bottom:1px solid #21262d; }}
  a {{ color:#58a6ff; }}
  .back-top {{ position:fixed; bottom:20px; right:20px; background:#21262d; color:#c9d1d9; border:none; border-radius:50%; width:44px; height:44px; font-size:20px; cursor:pointer; }}
</style>
</head>
<body>
<div class="container">

<h1>🏆 皐月賞 2026 全馬詳細診断レポート</h1>
<div class="meta">
  2026年4月19日（日）中山11R 芝2000m G1 18頭立て ／ 生成: {now}<br>
  過去13年（2013〜2025年）の皐月賞データをもとに8つの指標で各馬を評価。<br>
  総合スコアは各指標を0〜100点に正規化して重み付け加算。投資判断の参考資料としてご利用ください。
</div>

<div class="findings">
<h2>📊 過去データで判明した皐月賞の傾向</h2>
<ul>
  <li><strong>前走クラス×着順</strong>が最重要 ― G1前走1着 <strong>33.3%</strong> / G3前走1着 <strong>16.2%</strong> / G2前走1着はわずか <strong>4.0%</strong>（弥生賞1着も1.9%）</li>
  <li><strong>間隔・成長補正</strong> ― 9〜12週が <strong>13.9%</strong> で最高、13週以上も <strong>11.5%</strong>。5〜6週（スプリングS直行）はわずか <strong>1.8%</strong>。3歳馬は成長余地が大きい冬明け組が有利</li>
  <li><strong>脚質</strong> ― 中山は逃げ <strong>9.5%</strong> が突出。差しは3.2%、追込は0%。末脚自慢でも届かないコース</li>
  <li><strong>タイム指数ピーク</strong> ― 68以上は <strong>20%</strong>（今年は72.7のカヴァレリッツォのみ）。58〜63も <strong>8.0%</strong> と優秀</li>
  <li><strong>上り3F指数</strong> ― 60〜65が最適 <strong>8.3%</strong>。65以上は逆に <strong>2.8%</strong> と低下（末脚特化型は中山向きでない）</li>
  <li><strong>枠番</strong> ― 中枠(7〜12)が <strong>7.7%</strong>、外枠4.2%、内枠5.1%</li>
</ul>
</div>

{jk_html}

{and_html}

{elim_html_section}

<h2>📋 総合スコア サマリー（クリックで詳細へ）</h2>
<table class="summary-table">
  <thead>
    <tr>
      <th>総合Rnk</th><th>馬名</th><th>騎手</th><th>人気</th><th>単勝</th>
      <th>総合スコア</th>
      <th>AIモデル<br>35%</th><th>TI<br>30%</th><th>前走<br>クラス19%</th><th>騎手<br>10%</th><th>間隔<br>3%</th><th>上り3F<br>3%</th>
    </tr>
  </thead>
  <tbody>{summary_rows}</tbody>
</table>

<h2>🐎 全馬 個別詳細診断</h2>
{cards_html}

<h2>📐 指標の重み付け</h2>
<table class="weight-table">
  <thead><tr><th>指標</th><th>重み</th><th>根拠（過去皐月賞の勝率帯）</th></tr></thead>
  <tbody>
    <tr><td>AIモデル（距離/クラス）</td><td>35%</td><td>LightGBM LambdaMARTランカー順位。普段の予想モデルを軸に</td></tr>
    <tr><td>タイム指数ピーク</td><td>30%</td><td>68以上:20.0% / 63-68:3.8% / 58-63:8.0%（rho=0.23、グリッドサーチ最適）</td></tr>
    <tr><td>前走クラス×着順</td><td>19%</td><td>G1前走1着:33.3% / G3前走1着:16.2% / G2前走1着:4.0%（rho=0.21）</td></tr>
    <tr><td>騎手（全体勝率）</td><td>10%</td><td>rho=0.49だが逆因果（強い馬に良い騎手）割引・v3で縮小</td></tr>
    <tr><td>間隔・成長補正</td><td>3%</td><td>9-12週:13.9% / 13週以上:11.5% / 5-6週:1.8%（参考程度）</td></tr>
    <tr><td>上り3F指数</td><td>3%</td><td>60-65:8.3% / 55-60:5.6% / 65以上:2.8%（参考程度）</td></tr>
  </tbody>
</table>

<div style="color:#555;font-size:11px;margin-top:20px;line-height:2">
  ※ このレポートは統計的傾向の参考資料です。投資判断はご自身で行ってください。<br>
  ※ データソース：JRA-VAN / 過去13年（2013〜2025年）の皐月賞出走馬実績<br>
  ※ <a href="../satsuki_multi_2026.html">多角的分析サマリー版はこちら</a> ／ <a href="../satsuki_2026.html">前走データ特化MLモデルはこちら</a>
</div>

</div>
<button class="back-top" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑</button>
</body>
</html>"""

    out = ROOT / "docs/g1/satsuki_2026.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML出力: {out}")
    return out


if __name__ == "__main__":
    generate()
