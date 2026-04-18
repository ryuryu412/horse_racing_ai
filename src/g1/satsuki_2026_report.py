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
    "prep": 0.18, "ti": 0.16, "ur": 0.13, "model": 0.14,
    "interval": 0.13, "style": 0.10, "sire": 0.09, "waku": 0.04, "weight": 0.03,
}
NORM = {
    "prep": 33.3, "ti": 20.0, "ur": 8.3, "model": 1.0,
    "interval": 13.9, "style": 9.5, "sire": 33.3, "waku": 7.7, "weight": 7.6,
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

    # 前脚質（文字列）を正とする。前走脚質_numはコード体系が異なるため不使用
    style_txt = str(h.get("前脚質", "") or "").strip()
    style_map_txt = {"逃": (1, "逃げ"), "先": (2, "先行"), "中": (3, "差し(中団)"), "後": (4, "後方追込")}
    sn_int, st_l = style_map_txt.get(style_txt, (3, f"不明({style_txt})"))
    st_r = STYLE_RATE.get(sn_int, 3.2)
    s["style"] = st_r / NORM["style"]
    d["style"] = {"rate": st_r, "label": st_l, "max": 9.5}

    sire = str(h.get("種牡馬", "") or "")
    si_r = SIRE_RATE.get(sire, 3.0)
    s["sire"] = min(si_r / NORM["sire"], 1.0)
    d["sire"] = {"rate": si_r, "label": sire or "不明", "max": 33.3,
                 "notable": sire in SIRE_RATE}

    bnum = int(float(h.get("馬番", 1) or 1))
    if bnum <= 6:
        wg, wr = "内枠(1-6)", 5.1
    elif bnum <= 12:
        wg, wr = "中枠(7-12)", 7.7
    else:
        wg, wr = "外枠(13-18)", 4.2
    s["waku"] = wr / NORM["waku"]
    d["waku"] = {"rate": wr, "label": f"{bnum}番 {wg}", "max": 7.7}

    # 馬体重増減は当日計量後に確定するため現時点では使用不可
    bw_raw = h.get("馬体重", None)
    bw_available = bw_raw is not None and str(bw_raw) not in ("NaN", "nan", "", "None")
    if bw_available:
        wd = float(h.get("馬体重増減", 0) or 0)
        wt_r = lookup(wd, WEIGHT_RATE)
        bw_label = f"{int(wd):+d}kg（{bw_raw}kg）"
    else:
        wt_r = None  # 未確定
        bw_label = "⏳ 当日計量待ち"
    s["weight"] = (wt_r / NORM["weight"]) if wt_r is not None else None
    d["weight"] = {"rate": wt_r, "label": bw_label, "max": 7.6, "pending": wt_r is None}

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
        "prep": "前走クラス×着順", "ti": "タイム指数ピーク", "ur": "上り3F指数",
        "model": "AIモデル（距離/クラス）", "interval": "間隔・成長補正",
        "style": "脚質（中山適性）", "sire": "種牡馬", "waku": "枠番", "weight": "馬体重増減",
    }
    dims_html = "".join(dim_block(k, dim_labels[k], details[k], scores[k]) for k in dim_labels)

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

    # parquetの脚質: 0=逃,1=先,2=中,3=後 → STYLE_RATE key
    sn_raw = h.get("前走脚質_num", None)
    if sn_raw is not None and not (isinstance(sn_raw, float) and np.isnan(float(sn_raw) if sn_raw is not None else float("nan"))):
        sn = PARQUET_STYLE_MAP.get(safe_int(sn_raw, 2), 3)
    else:
        sn = 3
    s["style"] = STYLE_RATE.get(sn, 3.2) / NORM["style"]

    sire = str(h.get("種牡馬", "") or "")
    s["sire"] = min(SIRE_RATE.get(sire, 3.0) / NORM["sire"], 1.0)

    bnum = safe_int(h.get("馬番", 9), 9)
    wr = 5.1 if bnum <= 6 else 7.7 if bnum <= 12 else 4.2
    s["waku"] = wr / NORM["waku"]

    wd = safe_float(h.get("馬体重増減", 0))
    wt_r = lookup(wd, WEIGHT_RATE)
    s["weight"] = wt_r / NORM["weight"]

    total = sum(s[k] * WEIGHTS_NO_MODEL[k] for k in WEIGHTS_NO_MODEL) \
          / sum(WEIGHTS_NO_MODEL[k] for k in WEIGHTS_NO_MODEL) * 100
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
    eliminated = df_sc[~df_sc["top3"] & (df_sc["score"] < threshold)]

    return {
        "threshold": threshold,
        "top3_min": threshold,
        "top3_max": float(top3_df["score"].max()),
        "top3_mean": float(top3_df["score"].mean()),
        "all_df": df_sc,
        "eliminated_hist": eliminated,
        "total_horses": len(df_sc),
        "total_top3": len(top3_df),
    }


def elimination_section_html(thresh_data: dict, today_horses: list) -> str:
    t = thresh_data["threshold"]
    elim_today = [(item["h"]["馬名"], item["total_no_model"]) for item in today_horses
                  if item["total_no_model"] < t]
    near_today = [(item["h"]["馬名"], item["total_no_model"]) for item in today_horses
                  if t <= item["total_no_model"] < t + 5]

    # 消し馬リスト HTML
    if elim_today:
        elim_html = "".join(
            f'<div style="background:#1c0000;border:1px solid #5a1a1a;border-radius:6px;padding:10px 14px;margin-bottom:6px">'
            f'<span style="color:#e74c3c;font-weight:bold;font-size:14px">❌ {name}</span>'
            f'<span style="color:#555;font-size:12px;margin-left:12px">参照スコア {sc:.1f}（しきい値 {t:.1f} 未満）</span>'
            f'</div>'
            for name, sc in sorted(elim_today, key=lambda x: x[1])
        )
    else:
        elim_html = '<div style="color:#2ecc71;padding:10px">今年はしきい値を下回る馬なし（全馬が過去3着以内水準以上）</div>'

    near_html = ""
    if near_today:
        near_html = '<div style="color:#f1c40f;font-size:12px;margin-top:8px">▲ 際どいライン（+5pt以内）: ' + \
                    " / ".join(f"{n}（{s:.1f}）" for n, s in near_today) + '</div>'

    # 過去データ散布サマリー（年別トップ3最低スコア）
    df = thresh_data["all_df"]
    df["year"] = df["日付"].str[:4]
    year_min = df[df["top3"]].groupby("year")["score"].min().reset_index()
    year_rows = "".join(
        f'<tr><td style="text-align:center">{r.year}年</td>'
        f'<td style="text-align:center;color:{"#e74c3c" if r.score < 35 else "#f1c40f" if r.score < 45 else "#2ecc71"}">{r.score:.1f}</td></tr>'
        for _, r in year_rows_iter(year_min)
    )

    return f"""
<div style="background:#161b22;border-radius:10px;padding:20px;margin-bottom:24px;border-left:4px solid #e74c3c">
  <h2 style="color:#e74c3c;border:none;margin-top:0;padding-left:0">🚫 過去データによる消し馬分析</h2>
  <p style="color:#8b949e;font-size:12px;margin-bottom:16px;line-height:1.8">
    過去13年（2013〜2025）の皐月賞全出走馬（{thresh_data['total_horses']}頭）に同じスコアリングシステムを適用。<br>
    3着以内に入った馬の<strong style="color:#f0a500">最低スコアは {t:.1f}点</strong>（参照スコア = model次元除外の7指標）。<br>
    このしきい値を下回った馬は過去13年で一度も3着以内に入っていない。
  </p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;text-align:center">
    <div style="background:#1c2128;border-radius:6px;padding:12px">
      <div style="color:#e74c3c;font-size:22px;font-weight:bold">{t:.1f}</div>
      <div style="color:#8b949e;font-size:11px">3着以内最低スコア<br>（消しラインしきい値）</div>
    </div>
    <div style="background:#1c2128;border-radius:6px;padding:12px">
      <div style="color:#2ecc71;font-size:22px;font-weight:bold">{thresh_data['top3_mean']:.1f}</div>
      <div style="color:#8b949e;font-size:11px">3着以内平均スコア</div>
    </div>
    <div style="background:#1c2128;border-radius:6px;padding:12px">
      <div style="color:#f0a500;font-size:22px;font-weight:bold">{thresh_data['top3_max']:.1f}</div>
      <div style="color:#8b949e;font-size:11px">3着以内最高スコア</div>
    </div>
  </div>
  <h3 style="color:#e74c3c;font-size:14px;margin-bottom:10px">今年の消し馬候補（しきい値 {t:.1f}点未満）</h3>
  {elim_html}
  {near_html}
  <details style="margin-top:16px">
    <summary style="color:#8b949e;font-size:12px;cursor:pointer">▼ 年別3着以内最低スコア（過去13年）</summary>
    <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:12px">
      <thead><tr><th style="background:#21262d;padding:6px;text-align:center">年</th><th style="background:#21262d;padding:6px;text-align:center">3着以内最低スコア</th></tr></thead>
      <tbody>{year_rows}</tbody>
    </table>
  </details>
</div>"""


def year_rows_iter(year_min):
    yield from year_min.sort_values("year").iterrows()


def generate():
    print("データ読み込み中...")
    with open(ROOT / "data/raw/cache/出馬表形式4月19日オッズcsv.cache.pkl", "rb") as f:
        cache = pickle.load(f)
    result = cache["result"]
    r11 = result[(result["会場"] == "中") & (result["Ｒ"] == 11)].copy()

    print("過去皐月賞スコアリング（消し馬しきい値算出）...")
    thresh_data = compute_elimination_threshold()
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
          <td style="text-align:center">{'🟢' if sc['prep']>=0.7 else '🟡' if sc['prep']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['interval']>=0.7 else '🟡' if sc['interval']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['ti']>=0.7 else '🟡' if sc['ti']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['ur']>=0.7 else '🟡' if sc['ur']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['style']>=0.7 else '🟡' if sc['style']>=0.4 else '🔴'}</td>
          <td style="text-align:center">{'🟢' if sc['model']>=0.7 else '🟡' if sc['model']>=0.4 else '🔴'}</td>
          <td style="text-align:center;color:#555">{'⏳' if sc['weight'] is None else ('🟢' if sc['weight']>=0.7 else '🟡' if sc['weight']>=0.4 else '🔴')}</td>
        </tr>"""

    # 消し馬分析セクション
    elim_html_section = elimination_section_html(thresh_data, horses)

    # 馬別詳細カード
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
  <li><strong>前哨戦クラス×着順</strong>が最重要 ― G1前走1着 <strong>33.3%</strong> / G3前走1着 <strong>16.2%</strong> / G2前走1着はわずか <strong>4.0%</strong>（弥生賞1着も1.9%）</li>
  <li><strong>間隔・成長補正</strong> ― 9〜12週が <strong>13.9%</strong> で最高、13週以上も <strong>11.5%</strong>。5〜6週（スプリングS直行）はわずか <strong>1.8%</strong>。3歳馬は成長余地が大きい冬明け組が有利</li>
  <li><strong>脚質</strong> ― 中山は逃げ <strong>9.5%</strong> が突出。差しは3.2%、追込は0%。末脚自慢でも届かないコース</li>
  <li><strong>タイム指数ピーク</strong> ― 68以上は <strong>20%</strong>（今年は72.7のカヴァレリッツォのみ）。58〜63も <strong>8.0%</strong> と優秀</li>
  <li><strong>上り3F指数</strong> ― 60〜65が最適 <strong>8.3%</strong>。65以上は逆に <strong>2.8%</strong> と低下（末脚特化型は中山向きでない）</li>
  <li><strong>枠番</strong> ― 中枠(7〜12)が <strong>7.7%</strong>、外枠4.2%、内枠5.1%</li>
</ul>
</div>

{elim_html_section}

<h2>📋 総合スコア サマリー（クリックで詳細へ）</h2>
<table class="summary-table">
  <thead>
    <tr>
      <th>総合Rnk</th><th>馬名</th><th>騎手</th><th>人気</th><th>単勝</th><th>総合スコア</th>
      <th>前走<br>クラス×着順</th><th>間隔<br>成長補正</th><th>TI<br>ピーク</th><th>上り3F<br>指数</th><th>脚質</th><th>AIモデル</th>
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
    <tr><td>前走クラス×着順</td><td>18%</td><td>G1-1着:33.3% 〜 G2-1着:4.0%（最大差が大きく最重要）</td></tr>
    <tr><td>タイム指数ピーク</td><td>16%</td><td>68+:20.0% / 58-63:8.0% / 63-68:3.8%</td></tr>
    <tr><td>AIモデル（距離/クラス）</td><td>14%</td><td>過去走から学習したLightGBM LambdaMARTランカー順位</td></tr>
    <tr><td>間隔・成長補正</td><td>13%</td><td>9-12週:13.9% / 13週以上:11.5% / 5-6週:1.8%（3歳特有）</td></tr>
    <tr><td>上り3F指数</td><td>13%</td><td>60-65:8.3% / 55-60:5.6% / 65以上:2.8%（高すぎに注意）</td></tr>
    <tr><td>脚質（中山適性）</td><td>10%</td><td>逃げ:9.5% / 先行:4.6% / 差し:3.2% / 追込:0%</td></tr>
    <tr><td>種牡馬</td><td>9%</td><td>キタサンブラック:25% / ドレフォン:33% / ハーツクライ:0%(18頭で0勝)</td></tr>
    <tr><td>枠番</td><td>4%</td><td>中(7-12):7.7% / 内(1-6):5.1% / 外(13-18):4.2%</td></tr>
    <tr><td>馬体重増減</td><td>3%</td><td>維持(-4~+4):7.6% / 増加:4.5% / 減少:3.4%</td></tr>
  </tbody>
</table>

<div style="color:#555;font-size:11px;margin-top:20px;line-height:2">
  ※ このレポートは統計的傾向の参考資料です。投資判断はご自身で行ってください。<br>
  ※ データソース：JRA-VAN / 過去13年（2013〜2025年）の皐月賞出走馬実績<br>
  ※ <a href="../satsuki_multi_2026.html">多角的分析サマリー版はこちら</a> ／ <a href="../satsuki_2026.html">前哨戦特化MLモデルはこちら</a>
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
