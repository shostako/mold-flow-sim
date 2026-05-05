# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

`mold-flow-sim` は射出成形の流動解析を**極端に簡略化**した教育・概念検証用シミュレータ。
実機検討用ツールではない（READMEなし、テストなし、リンタ設定なし）。
モデル: Hele-Shaw近似 + Cross-WLF粘度 + Pseudo-Conduction法による疑似充填時間場。

## 実行コマンド

`.venv/` が用意済み。依存は `requirements.txt`。

```bash
# 依存セットアップ（未済の場合）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Streamlit UI（インタラクティブ）
streamlit run app.py

# CLI バッチデモ（パラメータスイープを outputs/<case>/ に出力）
python run_demo.py
python run_demo.py --out outputs --cases PP_baseline PP_dual_gate
```

`run_demo.py` の `cases` dict にハードコードされた7ケース（材料・温度・射出条件・ゲート数・圧縮成形）を回す。`--cases` で個別指定可。
出力は `outputs/<label>/{fill.gif, pressure.png, weld_airtraps.png, frames/}`。`outputs/` は gitignore 済み。

テスト・リンタ・型チェックはこのリポジトリには**存在しない**。追加する際は新規導入として扱え。

## アーキテクチャ

エントリポイントは2つ（`app.py` の Streamlit UI と `run_demo.py` の CLI）。両方とも `core/` パッケージを呼び出す。

### `core/` の責務分割

- **`materials.py`** — `MaterialDB`（`data/materials.json` から樹脂パラメータ読込）と Cross-WLF 粘度関数 `cross_wlf_viscosity(material, T_K, gamma_dot, P_Pa)`。代表せん断速度は `representative_shear_rate(V_mms, h_mm) = 6V/h`（Newtonian plate 近似）。
- **`geometry.py`** — `Geometry`（mask + thickness map + gates + cell_size_mm の dataclass）、`build_demo_geometry`（プレート+ランナー+スプルーの合成形状、ゲート数指定可）、`geometry_from_image`（画像から閾値処理でキャビティ抽出）。
- **`solver.py`** — `HeleShawSolver` と結果 dataclass `FlowResult`。中核アルゴリズムは下記。
- **`visualizer.py`** — `render_fill_animation`（GIF）、`render_pressure_map`、`render_weldlines`、`export_frames`（PNG連番）。matplotlib で `Agg` バックエンド固定。

### 中核アルゴリズム（`solver.py`）

充填過程を**時間ステップで進めない**。代わりに楕円型問題を一発解する：

```
∇·(S ∇τ) = 1   in cavity
τ = 0          at gates (Dirichlet)
S∇τ·n = 0      at walls (Neumann, no-flux)
S = h³ / (12·η_eff)   ← Hele-Shaw コンダクタンス
```

- `_build_linear_system` で5点ステンシル CSR を組み、`scipy.sparse.linalg.spsolve` で `τ` を解く。面コンダクタンスは隣接セルの**調和平均**。
- `τ` は擬似到達時間場。絶対時間化は `fill_time = (τ/τ_max) · (V_cavity/Q)`。
- `η_eff` はバルク温度 `0.7·T_melt + 0.3·T_mold` と代表せん断速度で Cross-WLF を1回評価する**定数値**（局所反復なし）。
- 圧縮成形 (`compression_molding=True`) は時間ステッピングではなく、`h` を `compression_factor` 倍に膨らませてコンダクタンスを上げ、`T_fill` を `compression_fraction/compression_factor + (1-compression_fraction)` で短縮する**等価モデル**。
- ウェルドライン: 8近傍中6個以上が自分より小さい `τ` を持つセル（合流リッジヒューリスティック）。
- エアトラップ: `τ` の局所最大点（最後に充填される位置）。

### 意図的にモデル化していないもの（変更禁止というより、追加実装する場合の設計判断ポイント）

- 過渡熱結合（金型壁での冷却・固化層形成）
- 真の3D流れ、ジェッティング、コーナー効果
- 結晶化・収縮・反り
- パッキング段階の保圧
- 局所せん断速度反復（粘度は単一代表値）

これらを「修正」しようとすると solver の前提が崩れる。新機能として別解法を足す方向で考えろ。

## データ

- `data/materials.json` — Cross-WLF パラメータ5樹脂（PP, ABS, PC, PA66, PMMA）。出典は generic 値であり、実プロジェクト用にはベンダー実測データを差し替える前提。
- `assets/` — 画像入力用の置き場（現状空）。
- `.codex` — 0バイトのマーカーファイル（おそらく外部ツール用）。

## UI と CLI の対応関係

`app.py` の sidebar ウィジェットは `HeleShawSolver` のコンストラクタ引数とほぼ1:1。`run_demo.py` の各 `case` dict も同じ引数を直接渡す。新パラメータを solver に足すなら両方に反映する必要がある。
