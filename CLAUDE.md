# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

`mold-flow-sim` は射出成形の流動解析を**極端に簡略化**した Python シミュレータ。
スコープは **B（初期スクリーニング）**：理想は商用CAE代替だが現状そこには程遠い。
モデル: Hele-Shaw近似 + Cross-WLF粘度 + Pseudo-Conduction法による疑似充填時間場。

公開リポジトリ: https://github.com/shostako/mold-flow-sim (MIT License)

## 実行コマンド

`.venv/` は **uv 製**（`uv 0.11.7`）。pip は入っていない。`uv pip install ...` を使え。

```bash
# 依存セットアップ
uv pip install -e ".[dev]" --python .venv/bin/python

# Streamlit UI（インタラクティブ）
streamlit run app.py

# CLI バッチデモ（パラメータスイープを outputs/<case>/ に出力）
python run_demo.py
python run_demo.py --out outputs --cases PP_baseline FilmGate_PP_default

# テスト・lint
.venv/bin/pytest tests/                        # 39 tests
.venv/bin/ruff check .                         # lint
.venv/bin/ruff format --check .                # format check（CI と同条件）
.venv/bin/ruff format .                        # format apply
```

`run_demo.py` のケース定義は2系統：
- `DEMO_CASES`（8件）— `build_demo_geometry` 系（プレート+ランナー+スプルー合成形状）。`PP_skin_layer` がスキン層モデル ON のサンプル
- `FILM_GATE_CASES`（4件）— `build_film_gate_geometry` 系（パラメトリックフィルムゲート、バランサー含む）

`--cases` はどちらの系列のキーも受け付ける。出力先は `outputs/<label>/{fill.gif, pressure.png, weld_airtraps.png, frames/}`、スキン層 ON ならさらに `skin.png / core.png` が追加される。`outputs/` は gitignore 済み。

## アーキテクチャ

エントリポイントは2つ（`app.py` Streamlit UI、`run_demo.py` CLI）。両方とも `core/` パッケージを呼ぶ。

### `core/` の責務分割

- **`materials.py`** — `MaterialDB`（`data/materials.json` から樹脂パラメータ読込）、`cross_wlf_viscosity(material, T_K, gamma_dot, P_Pa)`。代表せん断速度は `representative_shear_rate(V_mms, h_mm) = 6V/h`（Newtonian plate 近似）。
- **`geometry.py`** — `Geometry` データ容器、`build_demo_geometry`（合成プレート）、`build_film_gate_geometry` + `FilmGateConfig`（パラメトリックフィルムゲート、後述）、`geometry_from_image`（画像閾値処理）。
- **`solver.py`** — `HeleShawSolver` と結果 `FlowResult`。中核アルゴリズムは下記。
- **`visualizer.py`** — `render_fill_animation`（GIF）、`render_pressure_map`、`render_weldlines`、`export_frames`（PNG連番）。matplotlib で `Agg` バックエンド固定。

### 中核アルゴリズム（`solver.py`）

充填過程を**時間ステップで進めない**。代わりに楕円型問題を一発解する：

```
-∇·(S ∇τ) = 1   in cavity      ← 実装はこの符号（SPD form）
τ = 0           at gates (Dirichlet)
S∇τ·n = 0       at walls (Neumann, no-flux)
S = h³ / (12·η_eff)   ← Hele-Shaw コンダクタンス
```

**注意：solver.py の docstring（line 6付近）は `∇·(S∇τ) = 1` と書いているが、実装の離散化は `-∇·(S∇τ) = 1`（連続形では `∇·(S∇τ) = -1`）**。結果 τ は正しいが、docstring の符号が紛らわしい。これは `tests/test_solver_1d.py` のコメントで明文化されている。

- `_build_linear_system` で5点ステンシル CSR を組み、`scipy.sparse.linalg.spsolve` で `τ` を解く。面コンダクタンスは隣接セルの**調和平均**。**注意：行列組立がPython二重ループでN大に弱い**。中規模以上のメッシュでは性能ネックになる（vectorization 候補）。
- `τ` は擬似到達時間場。絶対時間化は `fill_time = (τ/τ_max) · (V_cavity/Q)`。
- `η_eff` はバルク温度 `0.7·T_melt + 0.3·T_mold` と代表せん断速度で Cross-WLF を1回評価する**定数値**（局所反復なし）。
- 圧縮成形 (`compression_molding=True`) は時間ステッピングではなく、`h` を `compression_factor` 倍に膨らませて、`T_fill` を `compression_fraction/compression_factor + (1-compression_fraction)` で短縮する**等価モデル**。
- ウェルドライン: 8近傍中6個以上が自分より小さい `τ` を持つセル（合流リッジヒューリスティック）。
- エアトラップ: `τ` の局所最大点（最後に充填されるセル）。

#### スキン層モデル（`skin_layer_enabled=True`）

壁面で樹脂が固化して育つスキン層を Stefan/Neumann 形で取り込む。コアのバルク温度低下は引き続き無視（Neumann 近似で熱結合を切離）。

```
s(t) = c_skin · √(α · t)              ← スキン厚さ [m]
h_core(x,y) = max(h - 2·s, h_min)     ← コア層（流動経路）
S = h_core³ / (12·η)                  ← Hele-Shaw コンダクタンス
```

- `α` (= `material.thermal_diffusivity_m2_s`) は樹脂の熱拡散率。`data/materials.json` に5樹脂分。代表値（PP=9e-8, ABS=1e-7, PC=1.3e-7, PA66=1e-7, PMMA=1.1e-7 m²/s）。
- `c_skin` (= `skin_growth_constant`) は無次元の成長定数。`0.0` で OFF と数値同一、`1.0` 付近が物理的代表値。
- `s` は τ に依存し、τ は `S(h_core)` に依存するので **fixed-point 反復**で釣り合わせる：1) baseline (`h_core=h`) で τ_baseline を解く → 2) `t_arrival = (τ/τ_max)·T_fill` から `s_new`、`h_core_new` を計算 → 3) 新しい `S` で τ を再解 → 4) `‖Δτ‖` が `skin_convergence_tol` を下回るか `skin_max_iterations` に達するまで反復。
- **絶対時間スケーリング**: 反復後の `τ_max` が baseline の `τ_max_0` に対して何倍に膨らんだかで `T_fill` を比例倍する（圧力一定近似 = 流量が抵抗増分だけ細る）。
- **short shot**: 反復後に `h - 2·s ≤ h_min` となったセルを `short_shot_mask` に記録。本来流路が遮断されるが、数値安定性のため `h_core` には `min_core_thickness_mm` 以上のフロアが残る。可視化で赤マーク。
- 出力: `FlowResult.skin_thickness_mm`, `core_thickness_mm`, `short_shot_mask`、metadata に `skin_iterations / skin_converged / T_fill_inflation / short_shot_cells / short_shot_fraction`。
- 可視化: `render_skin_layer_map(result, path)` でスキン厚マップ、`render_core_layer_map(result, path)` でコア層 + short shot。

### `build_film_gate_geometry` の形状仕様

長方形プレート + 「等脚台形 + 短辺を半円で置換」のランナー + 半円中心のバルブゲート（円形 Dirichlet）。

座標系（`y` 上向き、`x` 右向き）:

```
y_plate_top    = y_long_edge + Hp           ← 製品上端
y_long_edge    = y_short_edge + D           ← 製品下辺 = 台形長辺
y_flat_top     = y_short_edge + D_flat      ← 厚みスロープ開始
y_short_edge   = pad + d/2                  ← 半円中心 = 台形短辺 = バルブゲートのy
y_circle_bottom = pad                        ← 半円下端
```

**11パラメータ + バリデーション制約**:

| グループ | 記号 | 制約 |
|---------|------|------|
| プレート | `plate_w / plate_h / plate_thk` | — |
| ランナー上面投影 | `runner_long (L_long) / runner_short_diameter (d) / runner_depth (D)` | `L_long ≤ plate_w`, `L_long ≥ d` |
| ランナー肉厚 | `runner_thk (h_runner) / runner_flat_depth (D_flat) / runner_slope_depth (D_slope)` | `D_flat + D_slope = D` |
| バルブゲート | `valve_gate_diameter (d_valve)` | `d_valve ≤ d` |
| 接続 | `gate_width (W_gate)` | `W_gate ≤ L_long` |

**ゲート土手の実装**: 製品最下行のうち中央 `W_gate` 幅以外を `mask=False` に強制。台形最上行は `L_long` 全幅で残す（土手は製品側で形成）。

**厚みプロファイル**: 半円・台形 flat zone は `h_runner` 一定、台形 slope zone (`y_short + D_flat 〜 y_long`) は `h_runner → plate_thk` 線形補間、製品本体は `plate_thk`。

#### オプション機能：フローバランサー（▽ 肉盗み）

LGP（導光板）系の実機技術。バルブゲート1点からの放射状（楕円状）流動先端を、製品長辺全幅から均一充填に近づけるための**中央肉盗み**。

5パラメータ（`balancer_enabled=True` 時のみ有効）：

| パラメータ | 意味 | 制約 |
|----------|------|------|
| `balancer_enabled` | bool トグル | — |
| `balancer_base_width_mm` (W_bal) | ▽底辺幅（長辺側） | `≤ W_gate` |
| `balancer_height_mm` (H_bal) | ▽の頂点〜底辺距離 | apex がバルブゲート円を侵さないこと |
| `balancer_base_distance_from_gate_mm` | 底辺位置の半円中心からの y距離 | `≤ D` |
| `balancer_target_thickness_mm` (h_bal) | ▽内のキャビティ厚（一定） | `> 0`、通常 `= plate_thk` |

**実装**: ▽内のセルは `thickness_mm` を `h_bal` で**上書き**（既存 slope zone の補間値より小さい値で潰す）。`h_bal = plate_thk` のとき、▽内のキャビティ天井が**製品本体の天面と同じ高さ**になる。これが「製品平面と平行」（実機の肉盗み加工で底面を水平に削る形状）の数学表現。

**物理**: コンダクタンス `S = h³/(12η)` の `h` を ▽ 内で下げると、流路抵抗が `h³` で激増する。樹脂は ▽ を避けて両端側に回り込むため、製品下辺の左右端への到達タイミングが中央と揃う方向に補正される。

**設計の勘所**: `h_bal` を小さくする・`W_bal` を広げる・`H_bal/D` 比を上げると効果が強くなる。ただし極端にすると射出圧が跳ね上がる（実機）/ Pseudo-Conduction 解の数値条件が悪化する（このシミュレータ）。`run_demo.py` の `FilmGate_PP_balancer` がチューニング起点ケース。

### 意図的にモデル化していないもの

- コアのバルク温度低下と粘度の動的更新（スキン層は Stefan/Neumann 近似で扱うが、コア温度は melt のまま固定）
- 真の3D流れ、ジェッティング、コーナー効果
- 結晶化・収縮反り
- パッキング段階の保圧
- 局所せん断速度反復（粘度は単一代表値）
- 中立面メッシュ・非構造格子・STL/STEP 入力

これらを「solver に足す」のは前提が崩れる。新機能として別解法を立てて並列に置く方向で考えろ。

## テスト

`tests/` 配下に4ファイル、合計 **39テスト**：

- `test_smoke.py` — 4件: import / MaterialDB / build_demo_geometry / Cross-WLF 単調性
- `test_solver_1d.py` — 5件: 1Dストリップの解析解 `τ(x) = x(2L−x)/(2S)` との比較。max誤差 <2%、メッシュ細分化で誤差減少を保証
- `test_geometry_film_gate.py` — 24件: シルエット / 厚み / ゲート土手 / 体積スケール / バリデーション / バランサー / solver 統合
- `test_skin_layer.py` — 6件: skin OFF/ON、`c_skin=0` で baseline 復元、極薄肉での short shot 検出、metadata の整合性

新機能を足したら**該当する系統のテストファイルにテストを追加**するのが慣例。形状なら `test_geometry_*.py`、solver の挙動なら `test_solver_*.py` または `test_skin_layer.py`。

## 開発ワークフロー

このリポジトリは **feature branch + PR 運用**。直接 main に push するのはタイポ修正・ドキュメント微調整くらいに留める。

```bash
# feature 着手
git checkout -b feature/<topic>
# ... 実装 + テスト + ruff format + pytest ...
git push -u origin feature/<topic>
gh pr create --title "..." --body "..."

# CI green を確認してから
gh pr merge <N> --rebase --delete-branch
git checkout main && git pull
```

**push 前に必ず**:
- `ruff check .` （CI と同じ lint）
- `ruff format --check .` （format check は CI で別ステップ。`ruff check` だけでは検出されない）
- `pytest tests/` （39件全部 pass を確認）

CI 設定: `.github/workflows/ci.yml`。Python 3.11 / 3.12 マトリクスで上記3つを走らせる。Node.js 20 actions の deprecation warning が出るが2026-06-02までは無害。

## データ・依存

- `data/materials.json` — Cross-WLF パラメータ5樹脂（PP, ABS, PC, PA66, PMMA）。**generic 値**であり、実プロジェクト用にはベンダー実測データを差し替える前提。
- `assets/` — 画像入力用の置き場（現状空）。
- `.codex` — 0バイトの外部ツールマーカー（gitignore 済み）。
- 依存は `pyproject.toml` の `[project] dependencies` に集約。`requirements.txt` は廃止済み。

## UI と CLI の対応関係

`app.py` のサイドバー入力は3系統（`Demo plate` / `Film gate (parametric)` / `画像から生成`）。`run_demo.py` の `DEMO_CASES` / `FILM_GATE_CASES` も同じパラメータ群を扱う。**新パラメータを `HeleShawSolver` または `FilmGateConfig` に足すなら、UI と CLI の両方に反映する必要がある**。

特に `FilmGateConfig` の `D_flat + D_slope = D` 制約は、UI 側では「`D_flat / D` の比率スライダー」で表現してこの制約を自動満足させている（`app.py` の `flat_ratio` 変数）。CLI 側は直接 `D_flat` / `D_slope` を渡すので、case 定義時に和が `D` になることを手動で保証する必要がある。
