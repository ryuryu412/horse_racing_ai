"""
皐月賞 多角的分析レポート
8つの角度から各馬を評価して総合スコアを算出
"""
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

# -----------------------------------------------------------------------
# 過去の皐月賞から算出した各カテゴリの勝率テーブル
# -----------------------------------------------------------------------

# 前走クラス×着順 → 勝率
PREP_WIN_RATE = {
    (9, 1): 33.3, (9, 2): 0.0,  (9, 3): 0.0,
    (8, 1):  4.0, (8, 2): 4.5,  (8, 3): 0.0,
    (7, 1): 16.2, (7, 2): 16.7, (7, 3): 0.0,
    (6, 1):  0.0, (6, 2): 0.0,
}

# 枠番グループ → 勝率
WAKU_RATE = {"内(1-6)": 5.1, "中(7-12)": 7.7, "外(13-18)": 4.2}

# 脚質 → 勝率
STYLE_RATE = {1: 9.5, 2: 4.6, 3: 3.2, 4: 0.0}  # 逃,先,差,追
STYLE_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}

# タイム指数ピーク → 勝率
TI_PEAK_RATE = [
    (68, float("inf"), 20.0),
    (63, 68,  3.8),
    (58, 63,  8.0),
    (0,  58,  3.7),
]

# 上り3F指数ピーク → 勝率
UR_PEAK_RATE = [
    (65, float("inf"),  2.8),
    (60, 65,  8.3),
    (55, 60,  5.6),
    (0,  55,  2.9),
]

# 種牡馬 → 勝率
SIRE_RATE = {
    "キタサンブラック": 25.0,
    "ドレフォン":       33.3,
    "オルフェーヴル":   25.0,
    "フジキセキ":       25.0,
    "エピファネイア":   14.3,
    "ロードカナロア":   14.3,
    "キズナ":          11.1,
    "ディープインパクト": 8.8,
    "キングカメハメハ":  6.2,
}
SIRE_DEFAULT = 3.0

# 馬体重増減 → 勝率
WEIGHT_RATE = [
    (5,  float("inf"), 4.5),   # +5以上
    (-4, 5,            7.6),   # 維持
    (float("-inf"), -4, 3.4),  # -5以下
]

# 前走からの間隔（週）→ 勝率
# 9-12週が13.9%で最高。5-6週（スプリングS組）は1.8%で最低
# 3歳馬は長い間隔のほうが成長分が乗る傾向
INTERVAL_RATE = [
    (9,  13,  13.9),  # 9-12週: 共同通信杯→皐月等
    (13, 999, 11.5),  # 13週以上: ホープフルS明け等（成長余地大）
    (7,   9,   6.2),  # 7-8週
    (0,   5,   3.2),  # 4週以内（連闘気味）
    (5,   7,   1.8),  # 5-6週（スプリングS直行組）← 歴史的に最低
]

# -----------------------------------------------------------------------
# 正規化: 全カテゴリを0〜100点に換算するための最大値
# -----------------------------------------------------------------------
NORM = {
    "prep":     33.3,
    "waku":      7.7,
    "style":     9.5,
    "ti":       20.0,
    "ur":        8.3,
    "sire":     33.3,
    "weight":    7.6,
    "interval": 13.9,
    "model":     1.0,
}

WEIGHTS = {
    "prep":     0.18,
    "ti":       0.16,
    "ur":       0.13,
    "model":    0.14,
    "interval": 0.13,  # 成長・間隔補正
    "style":    0.10,
    "sire":     0.09,
    "waku":     0.04,
    "weight":   0.03,
}


def lookup_range(val, table):
    for lo, hi, rate in table:
        if lo <= val < hi:
            return rate
    return table[-1][2]


def score_horse(h: pd.Series) -> dict:
    scores = {}
    details = {}

    # 1. 前走クラス×着順
    cls = int(h.get("1走前_クラス_rank", 0) or 0)
    pos = int(h.get("1走前_着順_num", 99) or 99)
    prep_rate = PREP_WIN_RATE.get((cls, min(pos, 6)), PREP_WIN_RATE.get((cls, 1), 2.0))
    cls_label = {9:"G1",8:"G2",7:"G3",6:"OP",5:"3勝",4:"2勝",3:"1勝"}.get(cls,"-")
    scores["prep"] = min(prep_rate / NORM["prep"], 1.0)
    details["prep"] = f"{cls_label} {pos}着 → {prep_rate:.1f}%"

    # 2. 枠番
    bnum = h.get("馬番", 1) or 1
    if bnum <= 6:
        wg, wr = "内(1-6)", 5.1
    elif bnum <= 12:
        wg, wr = "中(7-12)", 7.7
    else:
        wg, wr = "外(13-18)", 4.2
    scores["waku"] = wr / NORM["waku"]
    details["waku"] = f"{wg} → {wr}%"

    # 3. 脚質
    style_num = h.get("前走脚質_num", np.nan)
    if pd.isna(style_num):
        sr = 5.0
        sl = "不明"
    else:
        style_num = int(style_num)
        sr = STYLE_RATE.get(style_num, 3.0)
        sl = STYLE_LABEL.get(style_num, "不明")
    scores["style"] = sr / NORM["style"]
    details["style"] = f"{sl} → {sr:.1f}%"

    # 4. タイム指数ピーク
    ti1 = h.get("1走前_タイム指数", 0) or 0
    ti2 = h.get("2走前_タイム指数", 0) or 0
    ti_peak = max(ti1, ti2)
    ti_rate = lookup_range(ti_peak, TI_PEAK_RATE)
    scores["ti"] = ti_rate / NORM["ti"]
    details["ti"] = f"peak {ti_peak:.1f} → {ti_rate:.1f}%"

    # 5. 上り3F指数ピーク
    ur1 = h.get("1走前_上り3F_指数", 0) or 0
    ur2 = h.get("2走前_上り3F_指数", 0) or 0
    ur_peak = max(ur1, ur2)
    ur_rate = lookup_range(ur_peak, UR_PEAK_RATE)
    scores["ur"] = ur_rate / NORM["ur"]
    details["ur"] = f"peak {ur_peak:.1f} → {ur_rate:.1f}%"

    # 6. 種牡馬
    sire = str(h.get("種牡馬", "") or "")
    sire_rate = SIRE_RATE.get(sire, SIRE_DEFAULT)
    scores["sire"] = min(sire_rate / NORM["sire"], 1.0)
    details["sire"] = f"{sire} → {sire_rate:.1f}%"

    # 7. 馬体重増減
    wdiff = h.get("馬体重増減", 0) or 0
    wrate = lookup_range(wdiff, WEIGHT_RATE)
    scores["weight"] = wrate / NORM["weight"]
    details["weight"] = f"{int(wdiff):+d}kg → {wrate:.1f}%"

    # 8. 間隔（3歳成長補正）
    interval = pd.to_numeric(h.get("間隔", np.nan), errors="coerce")
    if pd.isna(interval):
        irate = 5.6
        ilabel = "不明"
    else:
        interval = float(interval)
        irate = lookup_range(interval, INTERVAL_RATE)
        if interval >= 13:
            ilabel = f"{int(interval)}週 (冬明け・成長余地大)"
        elif interval >= 9:
            ilabel = f"{int(interval)}週 (適度な間隔)"
        elif interval >= 7:
            ilabel = f"{int(interval)}週 (中間隔)"
        elif interval >= 5:
            ilabel = f"{int(interval)}週 (スプリングS組)"
        else:
            ilabel = f"{int(interval)}週 (短間隔)"
    scores["interval"] = irate / NORM["interval"]
    details["interval"] = f"{ilabel} → {irate:.1f}%"

    # 9. 現行モデル（距離・クラス rankを統合）
    cur_r = h.get("cur_ランカー順位", 18) or 18
    sub_r = h.get("sub_ランカー順位", 18) or 18
    model_score = ((19 - cur_r) + (19 - sub_r)) / (2 * 18)
    scores["model"] = model_score
    details["model"] = f"距離{int(cur_r)}位 / クラス{int(sub_r)}位"

    # 総合スコア
    total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS) / sum(WEIGHTS.values())
    return {"total": total * 100, "scores": scores, "details": details}


def bar(val, max_val=100, width=80):
    """テキストバーグラフ用（HTML内でwidth%として使う）"""
    return min(val / max_val * 100, 100)


def score_color(v):
    if v >= 70:
        return "#2ecc71"
    elif v >= 50:
        return "#e67e22"
    elif v >= 30:
        return "#e74c3c"
    else:
        return "#555"


def dimension_cell(score_0to1, detail_str):
    pct = score_0to1 * 100
    color = score_color(pct)
    bar_w = int(min(pct, 100))
    return f"""<td>
      <div style="font-size:11px;color:#8b949e;margin-bottom:2px">{detail_str}</div>
      <div style="background:#21262d;border-radius:3px;height:6px;width:100%">
        <div style="background:{color};height:6px;border-radius:3px;width:{bar_w}%"></div>
      </div>
    </td>"""


def generate_html(rows: list) -> str:
    # 総合スコア順にソート
    rows_sorted = sorted(rows, key=lambda x: -x["total"])

    # トップ馬のリスト
    top3 = [r["name"] for r in rows_sorted[:3]]

    header_dims = ["前走<br>クラス×着順", "タイム指数<br>ピーク", "上り3F<br>指数", "通常モデル<br>(距離/クラス)",
                   "間隔<br>(成長補正)", "脚質", "種牡馬", "枠番", "馬体重<br>増減"]
    dim_keys    = ["prep", "ti", "ur", "model", "interval", "style", "sire", "waku", "weight"]

    rows_html = ""
    for i, r in enumerate(rows_sorted):
        rank = i + 1
        total = r["total"]
        if rank == 1:
            rbg = "background:#e74c3c;color:white"
        elif rank <= 3:
            rbg = "background:#e67e22;color:white"
        else:
            rbg = f"color:{score_color(total)}"

        dim_cells = "".join(
            dimension_cell(r["scores"][k], r["details"][k])
            for k in dim_keys
        )

        odds_str = f"{r['odds']:.1f}" if isinstance(r["odds"], float) and not np.isnan(r["odds"]) else "-"
        star = " ⭐" if rank <= 3 else ""

        rows_html += f"""
      <tr>
        <td style="text-align:center;font-weight:bold;font-size:15px;{rbg};padding:6px 4px">{rank}</td>
        <td style="font-weight:bold;font-size:13px;white-space:nowrap">{r['name']}{star}</td>
        <td style="color:#aaa;font-size:12px;white-space:nowrap">{r['jockey']}</td>
        <td style="color:#f0a500;font-weight:bold;text-align:center">{odds_str}</td>
        <td style="text-align:center">
          <span style="font-size:16px;font-weight:bold;color:{score_color(total)}">{total:.1f}</span>
          <div style="background:#21262d;border-radius:3px;height:5px;margin-top:3px">
            <div style="background:{score_color(total)};height:5px;border-radius:3px;width:{int(min(total,100))}%"></div>
          </div>
        </td>
        {dim_cells}
      </tr>"""

    # 凡例テーブル
    legend_rows = ""
    for key, label, data in [
        ("prep",     "前走クラス×着順",  "G1-1着:33.3% / G3-1着:16.2% / G2-1着:4.0% / G1-2着:0%"),
        ("ti",       "タイム指数ピーク", "68以上:20.0% / 58-63:8.0% / 63-68:3.8% / 58未満:3.7%"),
        ("ur",       "上り3F指数ピーク", "60-65:8.3% / 55-60:5.6% / 65以上:2.8% / 55未満:2.9%"),
        ("model",    "通常AIモデル",     "距離ランカー順位 + クラスランカー順位の統合"),
        ("interval", "間隔（成長補正）", "9-12週:13.9% / 13週以上:11.5% / 7-8週:6.2% / 4週以内:3.2% / 5-6週:1.8%"),
        ("style",    "脚質",            "逃げ:9.5% / 先行:4.6% / 差し:3.2% / 追込:0.0%（中山特有）"),
        ("sire",     "種牡馬",          "キタサンブラック25% / ドレフォン33% / G1血統優遇"),
        ("waku",     "枠番",            "中(7-12):7.7% / 内(1-6):5.1% / 外(13-18):4.2%"),
        ("weight",   "馬体重増減",      "維持(-4~+4):7.6% / 増加:4.5% / 減少:3.4%"),
    ]:
        w = int(WEIGHTS[key] * 100)
        legend_rows += f"<tr><td style='color:#c9d1d9'>{label}</td><td style='color:#f0a500;text-align:center'>{w}%</td><td style='color:#8b949e;font-size:12px'>{data}</td></tr>"

    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    th_style = "background:#21262d;color:#8b949e;padding:8px 6px;font-size:11px;text-align:center;white-space:nowrap"

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>皐月賞 多角的分析 2026</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Noto Sans JP',sans-serif; margin:0; padding:20px; }}
  .container {{ max-width:1400px; margin:0 auto; }}
  h1 {{ color:#f0a500; border-bottom:2px solid #f0a500; padding-bottom:8px; margin-bottom:4px; }}
  h2 {{ color:#8b949e; font-size:15px; margin:20px 0 8px; border-left:3px solid #f0a500; padding-left:8px; }}
  .subtitle {{ color:#8b949e; font-size:12px; margin-bottom:20px; }}
  table {{ width:100%; border-collapse:collapse; background:#161b22; border-radius:8px; overflow:hidden; font-size:12px; }}
  th {{ {th_style} }}
  td {{ padding:8px 6px; border-bottom:1px solid #21262d; vertical-align:middle; }}
  tr:hover {{ background:#1c2128; }}
  .legend-table {{ margin-top:12px; font-size:12px; }}
  .legend-table td {{ padding:5px 8px; }}
  .note {{ color:#555; font-size:11px; margin-top:16px; line-height:1.8; }}
  .findings {{ background:#161b22; border-radius:8px; padding:16px; margin-bottom:20px; }}
  .findings ul {{ margin:6px 0; padding-left:20px; color:#c9d1d9; font-size:13px; line-height:2; }}
  .tag {{ display:inline-block; background:#21262d; border-radius:4px; padding:1px 6px; font-size:11px; margin-left:4px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🏆 皐月賞 多角的分析レポート 2026</h1>
  <div class="subtitle">2026年4月19日（日）中山11R 芝2000m G1 ／ 生成: {now}<br>
  7つの角度から各馬を評価し、過去13年の皐月賞データをもとに重み付けスコアを算出</div>

  <div class="findings">
    <h2>📊 過去の皐月賞から見えた傾向</h2>
    <ul>
      <li><strong>前哨戦の質</strong>が最重要。<span style="color:#e74c3c">共同通信杯(G3)1着 → 勝率25%</span>、<span style="color:#e67e22">G1前走1着 → 33%</span>。弥生賞1着は意外に1.9%止まり。</li>
      <li><strong>間隔・成長補正</strong>：<span style="color:#2ecc71">9〜12週(13.9%)・13週以上(11.5%)</span>が高勝率。<span style="color:#e74c3c">5〜6週（スプリングS直行組）はわずか1.8%</span>。3歳の成長余地が大きい冬明け馬が有利。</li>
      <li><strong>脚質</strong>は中山らしく<span style="color:#2ecc71">逃げが9.5%</span>で最高。差しは3.2%、追込は0%。先行力が問われる。</li>
      <li><strong>タイム指数ピーク68以上</strong>は勝率20%。ただし今年はそのゾーンの馬がいない（最高値 72.7）ので相対比較が重要。</li>
      <li><strong>上り3F指数</strong>は60〜65が最適ゾーン(8.3%)。65超えは逆に2.8%（末脚型は中山向きでない）。</li>
      <li><strong>枠番</strong>は中枠(7-12)が優勢(7.7%)。内外は苦戦傾向。</li>
      <li><strong>種牡馬</strong>：キタサンブラック・オルフェーヴル系が高勝率。ハーツクライは頭数多いが勝率0%。</li>
    </ul>
  </div>

  <h2>🐎 馬別 総合スコアランキング</h2>
  <table>
    <thead>
      <tr>
        <th>総合Rnk</th><th>馬名</th><th>騎手</th><th>オッズ</th><th>総合<br>スコア</th>
        {''.join(f'<th>{h}</th>' for h in header_dims)}
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>

  <h2>📐 指標の重み付けと判定基準</h2>
  <table class="legend-table">
    <thead><tr><th>指標</th><th>重み</th><th>判定基準（過去皐月賞の勝率）</th></tr></thead>
    <tbody>{legend_rows}</tbody>
  </table>

  <div class="note">
    ※ 総合スコアは各指標を0〜100に正規化して重み付け加算したもの。絶対的な勝率予測ではなく「過去の傾向に合っているか」の総合評価。<br>
    ※ 前走クラス×着順は過去皐月賞の実績。G1前走とはホープフルS等の前走がG1に相当するケース。<br>
    ※ 脚質は前走脚質を参考値として使用。実際のレース展開次第で変動する。<br>
    ※ このレポートは統計的傾向の参考資料です。投資判断はご自身で行ってください。
  </div>
</div>
</body>
</html>"""


def main():
    print("皐月賞 多角的分析: データ読み込み中...")
    with open(ROOT / "data/raw/cache/出馬表形式4月19日オッズcsv.cache.pkl", "rb") as f:
        cache = pickle.load(f)
    result = cache["result"]
    r11 = result[(result["会場"] == "中") & (result["Ｒ"] == 11)].copy()

    print(f"出走馬: {len(r11)}頭")
    rows = []
    for _, h in r11.iterrows():
        sc = score_horse(h)
        odds = h.get("単勝オッズ", np.nan)
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            odds = np.nan
        rows.append({
            "name": h["馬名"],
            "jockey": h.get("騎手", "-"),
            "odds": odds,
            "total": sc["total"],
            "scores": sc["scores"],
            "details": sc["details"],
        })

    # コンソール出力
    print("\n===== 皐月賞 総合スコアランキング =====")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["total"]), 1):
        print(f"{i:2d}位 {r['name']:<14} 総合:{r['total']:5.1f}  "
              f"前走:{r['details']['prep']:<20}  "
              f"モデル:{r['details']['model']}")

    html = generate_html(rows)
    out = ROOT / "docs/satsuki_multi_2026.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nHTML出力: {out}")


if __name__ == "__main__":
    main()
