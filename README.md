# 競馬AI

JRA-VANのレースデータを使い、LightGBM + LambdaMARTランカーで単勝を予測するAI。
出馬表CSVを渡すだけで印（激熱/〇/▲/☆）付きのHTMLが出力される。

---

## 環境構築

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Python**: 3.11以上推奨
**OS**: Windows（JRA-VANアプリとのCOM連携のためWindows必須）

---

## ディレクトリ構成

```
horse_racing_ai/
├── src/                        # スクリプト本体
│   ├── 01_make_features.py     # 特徴量生成
│   ├── 02_train_model.py       # LightGBM学習
│   ├── 06_predict_from_card.py # ★実運用 出馬表予測
│   ├── 07_train_ranker.py      # LambdaMARTランカー学習
│   ├── 08_evaluate_models.py   # バックテスト・ROI評価
│   ├── 09_train_submodel.py    # サブモデル学習
│   ├── 11_train_class_ranker.py# クラスランカー学習
│   ├── weekly_update.py        # 週次マスターデータ更新
│   ├── _daily_roi_2026.py      # 予測ROI日別集計 → HTML
│   ├── _predict_time_roi_2026.py # 予想時点ROI集計 → HTML
│   └── archive/                # 旧スクリプト（参照用）
├── data/
│   ├── raw/
│   │   ├── master/             # JRA-VANマスターデータ（gitignore）
│   │   ├── cards/              # 出馬表形式CSV（週次）
│   │   ├── results/            # 結果確認CSV（週次）
│   │   └── cache/              # 予想時点キャッシュ（.cache.pkl）
│   ├── processed/              # 生成済み特徴量（gitignore）
│   ├── tohyo/                  # 実際の馬券データ
│   │   ├── all_tohyo.csv       # 統合済み馬券データ
│   │   └── archive/            # 日付別CSV原本
│   ├── merge_tohyo.py          # 馬券CSV統合スクリプト
│   └── update_roi_html.py      # 実馬券ROI HTML更新スクリプト
├── models_2025/                # 学習済みモデル（gitignore）
├── diary/devlog.md             # 開発日記
├── docs/                       # 仕様書・プロジェクトレポート
└── requirements.txt
```

---

## 毎週の予測フロー

### 1. 出馬表予測（レース前日〜当日）

```bash
python src/06_predict_from_card.py data/raw/cards/出馬表形式XX月XX日.csv
```

- 出力: `card_YYMMDD_YYYYMMDD_HHMM.html`（Gドライブ自動保存）
- ※ページの馬はレース直前にオッズを再確認してから購入判断

```bash
# HTMLのみ再生成（キャッシュ使用・約11秒）
python src/06_predict_from_card.py data/raw/cards/出馬表形式XX月XX日.csv --html-only
```

### 2. 馬券データ更新（毎週月曜）

```bash
# 1. 土日分の YYYYMMDD_tohyo.csv を data/tohyo/ にコピー
# 2. 統合（アーカイブへの移動も自動）
python data/merge_tohyo.py
# 3. 実馬券ROI HTML更新
python data/update_roi_html.py
# 4. GitHub Pages に公開
git add docs/actual_bet_roi.html
git commit -m "ROI更新 YYYYMMDD"
git push
```

- `all_tohyo.csv` は上書き更新されるので事前の退避不要
- アーカイブファイルを戻す作業も不要（差分のみ追記）
- push 後、数秒で GitHub Pages に反映される

### 3. ROI集計更新（結果確認CSVをもらったら）

```bash
# 結果確認CSVを data/raw/results/ に保存してから実行
python src/_daily_roi_2026.py       # 予測ROI（最終オッズ）
python src/_predict_time_roi_2026.py # 予測ROI（予想時点オッズ）
```

### 4. マスターデータ更新（週次）

```bash
# JRA-VANからrecent_all.csvをダウンロード → data/raw/master/ に上書き保存
python src/weekly_update.py
```

---

## 再学習フロー

```bash
python src/01_make_features.py          # 特徴量生成（約20分）
python src/02_train_model.py            # LightGBM学習（約30分）
python src/07_train_ranker.py           # ランカー学習
python src/09_train_submodel.py         # サブモデル学習
python src/11_train_class_ranker.py     # クラスランカー学習
python src/08_evaluate_models.py        # ROI評価確認
```

---

## 印ロジック（2026-03-28確定）

| 印 | 条件 | 単勝賭け金 | 2026実績ROI |
|---|---|---|---|
| 激熱 | 両モデル1位 & cur_diff≥10 & sub_diff≥10 & odds≥5 | 1,000円 | +253% |
| 〇 | 両モデル1位 & sub_diff≥10 & odds≥3 | 300円 | -24% |
| ▲ | 両Rnk≤2 & sub_diff≥10 & odds≥5 | 500円 | +91% |
| ☆ | 両Rnk≤3 & sub_diff≥10 & odds≥5 | 200円 | -26% |

**※印**: 能力条件クリアだがオッズ未達 → レース直前にオッズ再確認

---

## バックテスト結果（時系列後ろ20% / 約12.3万頭）

| 戦略 | 単勝ROI |
|---|---|
| ランカー1位 | +31.9% |
| ランカー1位 & 偏差値差+15以上 | +44.4% |
| ランカー1位 & 偏差値差+20以上 | +38.0% |

控除率25%の日本競馬で全戦略プラス。
2012年ホールドアウトテスト（10年以上前の未知データ）でもプラスを維持。

---

## 出力先

| ファイル | 内容 |
|---|---|
| https://ryuryu412.github.io/horse_racing_ai/actual_bet_roi.html | 実際の馬券ROI（スマホ対応・固定URL） |
| `G:/マイドライブ/競馬AI/ROI/actual_bet_roi.html` | 実際の馬券ROI（Gドライブ） |
| `G:/マイドライブ/競馬AI/daily_roi_2026.html` | 予測ROI日別（最終オッズ） |
| `G:/マイドライブ/競馬AI/predict_time_roi_2026.html` | 予測ROI（予想時点オッズ） |

---

## JV-Link API接続

```python
import win32com.client
jv = win32com.client.Dispatch('JVDTLab.JVLink')
ret = jv.JVInit('UNKNOWN')  # ← サービスキーは渡さない（v4.9系の仕様）
```

**注意**: `docs/` にあるJV-Link仕様書（PDF）は古い記述が含まれる。
JV-Link 4.9系では `JVInit()` の引数にサービスキーを渡す仕様が廃止されており、
サービスキーはJV-Link設定アプリ側で管理する。
仕様書の `JVInit(ServiceKey)` の記述は無視すること。

実行環境: **32bitのPython必須**（DLLがSysWow64のため）
```bash
py -3.12-32 -m venv .venv32
.venv32\Scripts\pip install pywin32
```

### データ取得の正しい手順

```python
# 必ずJVReadを呼ぶこと（呼ばないとltsが消費されてデータ再取得不可になる）
rc, readcnt, dldcnt, lts = jv.JVOpen('RACE', '20260101000000', 1, 0, 0, '')
if rc == 0 and readcnt > 0:
    records = []
    while True:
        rc2, buff, filename = jv.JVRead()
        if rc2 == 0:
            break
        records.append(buff)
    jv.JVClose()
```

### lts（最終取得タイムスタンプ）の仕組み
- JVOpenが成功するとltsがサーバー最新日時に更新される
- JVReadを呼ばずにJVCloseするだけでもltsが進む
- 以降、ltsより新しいデータがないと全スペック -303 になる
- **ltsはJVLinkAgentのメモリ管理 → PC再起動でリセット**
- PC再起動後は「JV-Link設定アプリ → 状態を取得する」で再認証が必要

### データ配信タイミング（目安）
| タイミング | 内容 |
|-----------|------|
| レース4〜5日前（水〜木） | 出馬表予備（SEPW） |
| レース前日（金〜土） | 出馬表確定（SEDW）・馬体重（WCWW） |
| レース当日 | リアルタイムオッズ（H1SW等） |
| レース翌日 | 成績確定データ |

### キャッシュ・データ保存先
- `C:\ProgramData\JRA-VAN\Data Lab\cache\` : 週次差分データ（JVD形式）
- `C:\ProgramData\JRA-VAN\Data Lab\data\` : 全期間履歴データ（1986年〜）
