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
.venv/bin/pytest tests/                        # 92 tests
.venv/bin/ruff check .                         # lint
.venv/bin/ruff format --check .                # format check（CI と同条件）
.venv/bin/ruff format .                        # format apply
```

`run_demo.py` のケース定義は3系統：
- `DEMO_CASES`（8件）— `build_demo_geometry` 系（プレート+ランナー+スプルー合成形状）。`PP_skin_layer` がスキン層モデル ON のサンプル。CLI 専用（UI からは選択不可）
- `FILM_GATE_CASES`（6件）— `build_film_gate_geometry` 系（パラメトリックフィルムゲート、バランサー含む）。`FilmGate_PP_stepped_stroke` は段差プレート + stroke 圧縮の参照ケース。`FilmGate_PP_multilayer_5L` は同じ段差プレートを **層別ソルバー (N=5 wall_refined)** で解いた対比ケース — gokuusu STEP4 議論ログの直接対応
- `DIRECT_GATE_CASES`（2件）— `build_direct_gate_geometry` 系（パラメトリックダイレクトゲート、Φピン+細スプルー+プレート）

`--cases` はどの系列のキーも受け付ける。出力先は `outputs/<label>/{fill.gif, pressure.png, weld_airtraps.png, frames/}`、スキン層 ON ならさらに `skin.png / core.png` が追加される。`outputs/` は gitignore 済み。

## アーキテクチャ

エントリポイントは2つ（`app.py` Streamlit UI、`run_demo.py` CLI）。両方とも `core/` パッケージを呼ぶ。

### `core/` の責務分割

- **`materials.py`** — `MaterialDB`（`data/materials.json` から樹脂パラメータ読込）、`cross_wlf_viscosity(material, T_K, gamma_dot, P_Pa)`。代表剪断速度は `representative_shear_rate(V_mms, h_mm) = 6V/h`（Newtonian plate 近似）。
- **`geometry.py`** — `Geometry` データ容器、`build_demo_geometry`（合成プレート、CLI 専用）、`build_film_gate_geometry` + `FilmGateConfig`（パラメトリックフィルムゲート、後述）、`build_direct_gate_geometry` + `DirectGateConfig`（パラメトリックダイレクトゲート、後述）。`Geometry.compression_mask` は **圧縮成形時にどのセルが膨らむか**を表す任意の bool 配列（`None` で全セル膨張＝旧挙動、配列指定で True セルだけ膨張）。パラメトリックビルダーは「製品本体だけ True」の compression_mask を埋め込む。
- **`solver.py`** — `HeleShawSolver` と結果 `FlowResult`。中核アルゴリズムは下記。**充填しないセルは充填時間を持たない**: スキン層が閉じたセル（`min_core_thickness_mm` で床打ちされる）と、それに封じ込められてゲートから到達できなくなった領域は `FlowResult.unfillable_mask` に入り、`fill_time_s` / `pressure_norm` は NaN になる。**未充填領域を切り離して τ を解き直し**（cavity 全体で解くと死んだ領域の体積が上流の生きたセルを通り、その τ が膨らむ — 実測3.3倍）、絶対時間の基準はその解の τ（`tau_max_flow`）から取る。ベースライン時間も充填する体積で計算する — 閉じたセルの τ は桁違いに大きく、基準に入れるとショートショットが「時間がかかる」に化けて T_fill が数倍に膨らむ。
- **`multilayer_solver.py`** — `MultilayerHeleShawSolver` と継承結果 `MultilayerFlowResult`。厚み方向を `N` 層に離散化して **層別温度・粘度** の coupling を fixed-point で解く新ソルバー。既存 `HeleShawSolver` を内部に保持して helper メソッド (`_effective_viscosity` / `_open_thickness_field` / `_solve_tau_field` / weld・airtrap) を委譲で再利用、線形代数の重複なし。詳細は後述「層別 Hele-Shaw ソルバー」セクション。
- **`multilayer_thermal.py`** — 純関数 `neumann_layer_temperatures()` と `poiseuille_shear_rates()`。前者は両壁から育つ Neumann 半無限解の重ね合わせ `T(z, t) = T_mold + (T_melt - T_mold) · [erf(z/(2√(αt))) + erf((h-z)/(2√(αt))) - 1]` (長時間極限で `T_mold` 以下に落ちるので clamp)、後者は Poiseuille 解析微分 `γ̇_k = (6V/h) · |2ζ - 1|` (中央発散回避の floor 付き)。両方とも shape `(N, ny, nx)` を返す。
- **`visualizer.py`** — `render_fill_animation`（GIF）、`render_pressure_map`、`render_weldlines`、`render_skin_layer_map` / `render_core_layer_map`（スキン層 ON 時のみ意味あり）、`render_layer_map` / `render_layer_grid` / `render_short_shot_map`（層別ソルバーの結果用、後述）、`export_frames`（PNG連番）。matplotlib で `Agg` バックエンド固定。**画像書き出し系**（PNG/GIF）はここ。**充填アニメの配色は `FILL_CMAP = "turbo"`**（`viridis` から変更）。理由は3つで、(1) 成形屋が読むのは絶対値でなく**等時線の形**（詰まる=遅い / ぶつかる=ウェルド / 途切れた先=最後）で、虹の色相コントラストがその縞を浮かせる（明度単調な viridis は逆に均す）、(2) **赤=最後に充填=リスク箇所**という意味論が一致する（viridis は上端が明るい黄色で逆）、(3) 商用 CAE と同じ配色なので相手に読み方の説明が要らない。`jet` でなく `turbo` なのは、jet がシアン/黄で明度が跳ねて**データに無い偽の縞**を描き、等時線と紛れるため。UI で turbo / jet / viridis / cividis を選べる（赤緑色覚への配慮は選択で担保）。**`_draw_isochrones`** が等時線を重ね、**`_fill_field_rgb` + `_unfilled_overlay` の2層描画**で「色は滑らか・輪郭はセル厳密」を両立する — 色の層は `_nearest_extend` でキャビティ外まで延長した不透明 RGBA を `bilinear` で描き、その上から未充填セルを `nearest` の不透明オーバーレイで塗り潰す。**充填時間は連続場なので色の補間は嘘にならないが、輪郭と溶融先端は mask のとおりが正しい**ので、アルファに mask を持たせて補間する（＝先端が半セル滲む）ことは避けている。**肉厚場は本質的に不連続**（t0.35 と t0.50 の間に中間肉厚は実在しない）なので、この平滑化を肉厚マップへ横展開してはいけない — 3D の flat-top 判断と同じ原則。**肉厚マップの配色は `THICKNESS_CMAP = "cividis_r"`**（`cividis` から反転）。肉厚は解いた結果ではなく**入力＝設計者が描いた形状**なので、結果図の虹ではなく単一量の明度 ramp を当てて一目で別カテゴリに見せる。向きは**明＝薄肉 / 暗＝厚肉**で、「インクの濃さ＝物質の量」という普遍的な直観に合うほか、透明樹脂の成形品では**厚いほど光が減衰して実際に暗く見える**（LGP を見慣れた目には反転前が現実と逆）。単色系（`Blues` / `bone_r`）でなく `cividis_r` なのは、**薄肉端が彩度を保つ必要がある**ため — 製品プレートは最薄領域であると同時に唯一注視される領域で、低端が白に寄るマップは製品を白飛びさせ、隣接ゾーン間の段差コントラストを潰し、3D では天面が薄グレーの PL 床と白背景に溶ける（`bone_r` で実測、0.35mm 帯が純白になり外形が消えた）。加えて cividis は色覚多様性向けに設計されており、その性質は反転しても残る。

  **配色ガードの組み方**（`tests/colorimetry.py` に色の計算を集約、`tests/test_visualizer_3d.py` / `test_visualizer_layer.py` が使う）: 配色は名前で照合せず**実際に描かれる ramp を測色して**判定する。名前だけの比較は `Cividis_r` が実は反転していなくても通る。(1) **単調性** — ramp を stop 間まで密サンプルして隣接サンプルが必ず暗くなること。両端だけの判定は少なからぬ組込配色に破られる（plotly 6.7.0 実測で 188ramp 中 26本。`plotly>=5.18` は上限無しなので件数は不変条件ではない）。最悪の `jet_r` は両端の輝度差 0.070 に対し中間で 0.719 逆行する＝「明→暗」を名乗るレインボー。密サンプルが要るのは Plotly が符号化 sRGB で線形補間する一方 relative luminance が非線形変換を先に掛けるためで、伝達関数が凸なので、区間内の輝度は暗い側の端点より下に沈んでから戻りうる — **これは色空間の性質であって特定の配色の性質ではない**。`bluered_r` が今それを示している（0.0023）が、証人は構成した反例（緑→マゼンタ）で固定してある。(2) **可読性** — 薄肉端と厚肉端のコントラスト比 ≥ 3:1（WCAG 1.4.11）。単調なだけなら近黒→黒の ramp も通り、順序は復元できても地図が読めない。(3) **白との分離** — 薄肉端の HSV 彩度 > 0.5。`cividis_r` の黄は白に対し輝度では 1.25:1 しかなく、**白背景から分離しているのは彩度であって明度ではない**ので、明度で判定すると採用した配色自身を弾く。輝度は必ず sRGB を線形化してから求める — ガンマ符号化のまま加重和を取るのは luma であって relative luminance ではなく、`rgb(205,78,46)→rgb(242,34,36)` のように符号が食い違う。
- **`visualizer_3d.py`** — `render_3d_thickness_map` / `render_3d_fill_time` / `render_3d_pressure`。**インタラクティブな3D表示**用。Plotly の `go.Figure` を返し、Streamlit の `st.plotly_chart` で埋め込む想定。各図は **PL（Z=0）= 半透明の薄グレー床 + 側壁 + 天面（Z=h、物理量で着色）** の3トレース構成で、**3トレースとも cavity セルだけを張る疎な `go.Mesh3d`**（天面・側壁は `coloraxis="coloraxis"` 共有、1本のカラーバー）。**各 cavity セルを「厚み一定の平らな天面ブロック」として描く**（flat-top）：`_cavity_corner_mesh` がセル端（隅）を頂点にした flat-top quad を組み、**同一厚みの隣接セルは隅を共有**して頂点を約1×（cavity セル数相当）に抑える（厚みが違うセルは隅を分けて段差の隙間を作る／`np.unique` でベクトル化、Python per-cell ループ無し）。天面の着色は**面ごと**（`intensitymode="cell"`、各セル＝平らな1パッチ＝2D マップと一致）。**厚みの段差はこの flat-top＋段差壁で「くっきりした縦の段差」として描かれる**（セル中心補間だと段差が1セル幅の傾斜に均されてしまうのを回避）——`_build_side_walls` が境界壁（cavity 端、PL→厚み）に加えて、隣接セルの厚みが変わる内部辺に**段差壁**（2つの厚みを繋ぐ縦 quad、+x/+y 方向のみで各辺1回・owner=高い側）を立てる。全 cavity セルが個別 quad を持つので**境界の侵食はゼロ**（旧セル中心三角形化が塞げなかった3セル角・1セル幅も完全カバー）。**【意図したトレードオフ】** 厚み差を全部「段差」扱いするので、**意図的に連続なテーパ**（`build_film_gate_geometry` のランナースロープ等＝多セルで補間）は滑らかなランプでなく微細な階段に描かれる。設計段差（1セルで Δ≈0.15）と連続勾配（≈0.15/セル）は**1セル差が同値で magnitude では区別不能**、唯一の堅牢な信号（局所曲率/二階差分）はランプ端で脆く表示専用機能には過剰なため、製品面の段差（プレート分割・バランサー）をくっきり出すことを優先して flat-top を選択（階段化するテーパは流路側＝離散化データそのままの正直な描画）。Codex P2 を承知の上で維持。**天面・床は元々フルグリッドの `go.Surface` だったが**、Surface は cavity 外も NaN で全格子を持つうえグリッド／ライティング機構を抱える重トレースで WebGL 回転が重く、疎 Mesh3d 化で回転負荷を大幅減。肉厚図の配色は `THICKNESS_COLORSCALE = "Cividis_r"`＝2D の `THICKNESS_CMAP` と同じ ramp で、設計図と立体図が「厚い」の見た目で食い違わないようにしてある（Plotly は名前を大文字化するので定数は別持ち、整合は `tests/test_visualizer_3d.py` が担保）。**`aspectmode="data"`** で x/y/z すべて mm 等倍（誇張なし）。プレートが薄板に見えるのは実物比率そのもの。物理は 2D Hele-Shaw のまま、表現上の3D化のみ。**3D は常にネイティブ解像度**（解析メッシュそのまま）で描画する。かつて「表示専用の解析的精細化」（外形を `cell_size/k` で再ラスタ化して 3D だけ細かくする `refine_for_display` 機能, PR #41）を入れたが、**疎 Mesh3d 化しても精細化時の描画が重く、効果に見合わないとして撤回**（精細化機構は削除、Mesh3d 化は維持）。3D を細かくしたいときは**解析メッシュ自体を下げる**（解析が重くなる代わりに 2D 結果も 3D も細かくなる）運用。

### 中核アルゴリズム（`solver.py`）

充填過程を**時間ステップで進めない**。代わりに楕円型問題を一発解する：

```
-∇·(S ∇τ) = 1   in cavity      ← 実装はこの符号（SPD form）
τ = 0           at gates (Dirichlet)
S∇τ·n = 0       at walls (Neumann, no-flux)
S = h³ / (12·η_eff)   ← Hele-Shaw コンダクタンス
```

符号は `solver.py` の docstring と `README.md` も**この形に揃えてある**（v0.24.0 で統一）。以前は両方が `∇·(S∇τ) = 1` と書いていて実装と食い違っていた。結果 τ は正しかったが、読む側が符号を追うたびに引っかかる。連続形で書くなら `∇·(S∇τ) = -1`。

- `_build_linear_system` で5点ステンシル CSR を組み、`scipy.sparse.linalg.spsolve` で `τ` を解く。面コンダクタンスは隣接セルの**調和平均**。**注意：行列組立がPython二重ループでN大に弱い**。中規模以上のメッシュでは性能ネックになる（vectorization 候補）。
- `τ` は擬似到達時間場。絶対時間化は `fill_time = (τ/τ_max) · (V_cavity/Q)`。
- `η_eff` はバルク温度 `0.7·T_melt + 0.3·T_mold` と代表剪断速度で Cross-WLF を1回評価する**定数値**（局所反復なし）。
- 圧縮成形 (`compression_molding=True`) は時間ステッピングではなく、`h` を膨らませて `T_fill` を `compression_fraction/effective_factor + (1-compression_fraction)` で短縮する**等価モデル**。膨張対象は `compression_mask & mask` のセル（None なら全 cavity）。**2 モード対応**：
  - **factor モード**（デフォルト、後方互換）: `h_eff = h * compression_factor`。`effective_factor = 1 + (compression_factor − 1) · f_comp`、`f_comp = Geometry.compression_volume_fraction()`（圧縮対象セル**体積** / 全 cavity 体積）。同じ倍率を全 target セルに掛けるので**薄肉ほど絶対膨張量が大きい**。圧縮比（型開き量比）が設計指標のときに使う。
  - **stroke モード**（`compression_stroke_mm` が None 以外）: `h_eff = h + stroke`。全 target セルに同じ絶対量を加算するので**段差プレートの段差（例: t0.50 − t0.35 = 0.15 mm）が圧縮位相中も保存される**。金型シム量（絶対距離）が設計指標のときに使う。`effective_factor = 1 + stroke · A_cm / V_total`、`A_cm = Geometry.compression_area_mm2()`（圧縮対象セルの**面積** mm²）、`V_total = volume_cm3 × 1000` mm³。
  - どちらのモードでも Film gate / Direct gate のように**プレート本体だけが膨らむ**形状ではランナー・スプルーが膨張に寄与しない分 `effective_factor` が薄まる（実機の挙動と整合）。uniform プレートで `factor = (h + stroke)/h` に揃えると両モードは厳密に等価（`tests/test_compression_stroke.py::test_uniform_plate_stroke_factor_equivalence_for_T_fill` で担保）。
- ウェルドライン: 8近傍中6個以上が自分より小さい `τ` を持つセル（合流リッジヒューリスティック）。
- エアトラップ: `τ` の局所最大点（最後に充填されるセル）。

#### スキン層モデル（`skin_layer_enabled=True`）

壁面で樹脂が固化して育つスキン層を Stefan/Neumann 形で取り込む。コアのバルク温度低下は引き続き無視（Neumann 近似で熱結合を切離）。

```
s(t) = c_skin · √(α · t)              ← スキン厚さ [m]
h_core(x,y) = max(h - 2·s, h_min)     ← コア層（流動経路）
S = h_core³ / (12·η)                  ← Hele-Shaw コンダクタンス
```

- `α` (= `material.thermal_diffusivity_m2_s`) は樹脂の熱拡散率。`data/materials.json` に8樹脂分。代表値（PP=9e-8, PP_T10=1.05e-7, PP_T20=1.30e-7, PP_T30=1.55e-7, ABS=1e-7, PC=1.3e-7, PA66=1e-7, PMMA=1.1e-7 m²/s）。タルク強化系は熱伝導率がベース PP より高いので α は単調増加。
- `c_skin` (= `skin_growth_constant`) は無次元の成長定数。`0.0` で OFF と数値同一、`1.0` 付近が物理的代表値。
- `s` は τ に依存し、τ は `S(h_core)` に依存するので **fixed-point 反復**で釣り合わせる：1) baseline (`h_core=h`) で τ_baseline を解く → 2) `t_arrival = (τ/τ_max)·T_fill` から `s_new`、`h_core_new` を計算 → 3) 新しい `S` で τ を再解 → 4) `‖Δτ‖` が `skin_convergence_tol` を下回るか `skin_max_iterations` に達するまで反復。
- **絶対時間スケーリング**: 反復後の `τ_max` が baseline の `τ_max_0` に対して何倍に膨らんだかで `T_fill` を比例倍する（圧力一定近似 = 流量が抵抗増分だけ細る）。
- **short shot**: 反復後に `h - 2·s ≤ h_min` となったセルを `short_shot_mask` に記録。本来流路が遮断されるが、数値安定性のため `h_core` には `min_core_thickness_mm` 以上のフロアが残る。可視化で赤マーク。
- 出力: `FlowResult.skin_thickness_mm`, `core_thickness_mm`, `short_shot_mask`、metadata に `skin_iterations / skin_converged / T_fill_inflation / short_shot_cells / short_shot_fraction`。
- 可視化: `render_skin_layer_map(result, path)` でスキン厚マップ、`render_core_layer_map(result, path)` でコア層 + short shot。

#### 層別 Hele-Shaw ソルバー（`MultilayerHeleShawSolver`）

スキン層モデルが「壁面凍結フロント」だけを扱うのに対し、こちらは厚み方向を `N` 層に離散化し、各層に **温度・粘度・剪断速度** を持たせる完全結合ソルバー。`core/multilayer_solver.py` に独立クラスとして実装、既存 `HeleShawSolver` には一切手を入れない (内部で保持して helper メソッドを委譲する形)。スキン層モデルとは排他的に使う想定 (UI ラジオで強制、CLI `_solve_and_export` で `skin_layer and multilayer` を `ValueError`)。

**厚み離散化**:

```
ζ_k ∈ [0, 1]                 ← 相対厚み座標 (N+1 個の境界)
h_k(x,y) = (ζ_k - ζ_{k-1}) · h_total(x,y)   ← 絶対層厚 (圧縮 factor/stroke の効果は h_total に統合済み)
```

- `uniform`: `ζ_k = k/N`
- `wall_refined` (デフォルト): Chebyshev-Lobatto 点 `ζ_k = 0.5·(1 - cos(πk/N))`。Neumann 勾配の急な壁面で解像度を稼ぐ (N=6 で `[0, 0.067, 0.25, 0.5, 0.75, 0.933, 1]`)。N=1 は uniform にフォールバック (Hele-Shaw 単層極限を維持)。

**Poiseuille モーメント積分** (`_multilayer_conductance`):

```
S_total(x,y) = (h_total³ / 2) · Σ_k m_k / η_k
m_k = [ζ²/2 - ζ³/3]_{ζ_{k-1}}^{ζ_k}
Σ_k m_k = 1/6   ← Hele-Shaw 因子 (どの distribution でも保存)
```

N=1 で `m_1 = 1/6` → `S = h³/(12η)` と厳密に一致 (`test_n1_matches_legacy_tau` で担保)。

**Fixed-point 結合** (`thermal_coupling=True` のみ):

```
初期化: τ ← baseline (等温・代表 η) で _solve_tau_field
反復:
  t_arr ← (τ/τ_max) · T_fill
  T_k(x,y) ← neumann_layer_temperatures(ζ_centers, t_arr, h_total, T_melt, T_mold, α)  (max(_, T_mold) で clamp)
  γ̇_k(x,y) ← poiseuille_shear_rates(ζ_centers, V, h_total, floor=0.01)
  η_k(x,y) ← cross_wlf_viscosity(material, T_k, γ̇_k, 0)
  S_total ← Σ_k h_total³ m_k / (12 η_k)
  τ_new ← _solve_tau_field(S_total, dirichlet)
  T_fill_new ← T_fill_baseline · (τ_max_new / τ_max_baseline)   ← 圧力一定近似
  rel ← ‖τ_new - τ‖_2 / ‖τ‖_2
  if rel >= prev_rel: τ_new ← (1-ω)τ_old + ω·τ_new (適応的 damping、ω=damping_factor=0.7 既定)
  if rel < convergence_tol: 収束、終了
```

- `max_iterations = 8`、`convergence_tol = 1e-3` 既定 (スキン層モデルより 1〜2 回深め、温度の変化が緩いため)。
- `shear_rate_floor_factor = 0.01` で中央層 γ̇ をクリップ (Cross-WLF ゼロ剪断粘度 `D₁` の暴走を防止)。

**ショートショット判定** (PR-C):

中央層 (`k = N // 2`) の最終温度が固化しきい値を下回るセルを `short_shot_mask` にマーク:

```
T_solid = T_mold + solidification_temperature_fraction · (T_melt - T_mold)
short_shot_mask = cavity & (T_k_mid <= T_solid)
```

- `solidification_temperature_fraction = 0.3` 既定 (PP 想定の粗い目安、Tg 近傍)
- 材料 DB に固化温度フィールドは追加せず、既存 `T_melt_recommended` / `T_mold_recommended` から派生

**剪断発熱 (viscous dissipation) — 段階1**:

`shear_heating_enabled=True` で Neumann 温度に粘性散逸補正を加算する閉形式モデル：

```
ΔT_shear,k(x,y) = (η_k · γ̇_k²) / (ρ · cp) · min(t_arr(x,y), τ_thermal(x,y))
τ_thermal(x,y) = h_total(x,y)² / (π² · α)         ← 厚み方向 1D 熱拡散の最低モード時定数
T_k_corrected = T_Neumann,k + ΔT_shear,k         ← 局所温度上昇
```

fixed-point ループの**各反復で 1 回**評価する：前イテレーションの `η_k` で発熱密度を計算 → ΔT_shear,k を Neumann 温度に加算 → 新しい T_k で Cross-WLF を再評価 → η_k 更新 →（負のフィードバック: T↑ → η↓ → 発熱↓）→ 全体が同じ fixed-point に収束。物理的には保証された定常解ではないが、**Br ≈ 1 までの領域では妥当**な近似。

Brinkman 数 `Br = η·γ̇²·h² / (k·ΔT_ref)` (`ΔT_ref = T_melt − T_mold`) は `shear_heating_enabled` の値に関わらず**毎回計算してメタデータに出す**。`Br < 0.5` で発熱無視可、`Br > 2` で発熱支配 → 段階2 (1D FDM 陰解法) が本来必要なシグナル。

熱伝導率は材料 DB に保持せず、`k = α · ρ · cp` で派生する (PP で約 0.16 W/(m·K))。新規フィールドは `specific_heat_J_kgK` のみ追加 (8 樹脂分)。

**出力**:

- `MultilayerFlowResult(FlowResult)` (継承): 既存フィールドに加えて `layer_thickness_mm` / `layer_temperature_K` / `layer_viscosity_Pa_s_field` / `layer_shear_rate_s_inv` / `layer_shear_heating_dT_K` / `layer_brinkman_number` (shape `(N, ny, nx)`)。`thermal_coupling=False` では `layer_temperature_K` 以下すべて None (thickness のみ populated)。
- `short_shot_mask` (継承元 FlowResult のフィールド) が中央層温度ベースで populated。剪断発熱補正後の温度を反映。
- `metadata` に `solver_kind="multilayer"` / `num_layers` / `layer_distribution` / `layer_zeta` / `layer_moments` / `multilayer_iterations` / `multilayer_converged` / `T_fill_inflation` / `damping_factor` / `damping_events` / `T_solid_K` / `short_shot_cells` / `short_shot_fraction` / `shear_heating_enabled` / `shear_heating_max_K` / `shear_heating_mean_K` / `brinkman_number_max` / `brinkman_number_mean` / `specific_heat_J_kgK` / `thermal_conductivity_W_mK`。

**可視化** (`core/visualizer.py`):

- `render_layer_map(result, layer_idx, path, field)` — 1 層単独マップ。`field` は `"temperature"` / `"viscosity"` / `"shear_rate"` / `"thickness"`。viscosity と shear_rate は自動で log カラースケール。
- `render_layer_grid(result, path, field)` — 全 N 層を 1 PNG にタイル化、共通スケールで壁→中央の勾配が一目で読める。
- `render_short_shot_map(result, path)` — ショートショットセルを赤マークでオーバーレイ。flagged 0 時は "no short shot" 注釈。

**重要な限界**:

層別化しても **2D Hele-Shaw の前提（面内のみ圧力勾配、厚み方向は積分）は不変**。捕捉できない物理：

- **面内コーナー効果**: 角部の渦、二次流、流動偏向。Hele-Shaw 近似では原理的に出ない。
- **ジェッティング**: 高速ゲートで樹脂が固相のまま射出される現象。3D 流れ前提。
- **コア対流項**: 厚み方向の対流による熱輸送。1D Neumann は純粋拡散のみ。
- **剪断発熱の段階1近似**: `min(t_arr, τ_thermal)` で頭打ちにする閉形式は **Br > 2 で系統的にズレる** (発熱と熱伝導のバランスをエネルギー的に保証していない)。Br ≫ 1 領域の精密予測には**段階2 (層別 1D FDM 陰解法)** が必要。
- これらは **完全 3D FVM / FEM ソルバー (別ロードマップ)** が必要。

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

**11パラメータ + プレート分割3パラメータ + バリデーション制約**:

| グループ | 記号 | 制約 |
|---------|------|------|
| プレート | `plate_w / plate_h / plate_thk` | — |
| ランナー上面投影 | `runner_long (L_long) / runner_short_diameter (d) / runner_depth (D)` | `L_long ≤ plate_w`, `L_long ≥ d` |
| ランナー肉厚 | `runner_thk (h_runner) / runner_flat_depth (D_flat) / runner_slope_depth (D_slope)` | `D_flat + D_slope = D` |
| バルブゲート | `valve_gate_diameter (d_valve)` | `d_valve ≤ d` |
| 接続 | `gate_width (W_gate)` | `W_gate ≤ L_long` |
| プレート分割 | `plate_split_height_mm / plate_lower_thk_mm / plate_upper_thk_mm` | `0 ≤ split ≤ Hp`、`0` で均一モード |

**ゲート土手の実装**: 製品最下行のうち中央 `W_gate` 幅以外を `mask=False` に強制。台形最上行は `L_long` 全幅で残す（土手は製品側で形成）。

**厚みプロファイル**: 半円・台形 flat zone は `h_runner` 一定、台形 slope zone (`y_short + D_flat 〜 y_long`) は `h_runner → plate_lower_thk` 線形補間、製品本体は ゲート側 `[y_long, y_long + split]` が `plate_lower_thk`、反ゲート側 `[y_long + split, y_plate_top]` が `plate_upper_thk`。`split = 0` の均一モードでは `plate_lower_thk = plate_upper_thk = plate_thk`、slope は `plate_thk` に補間して従来挙動と一致。

#### オプション機能：フローバランサー（▽ 肉盗み）

LGP（導光板）系の実機技術。バルブゲート1点からの放射状（楕円状）流動先端を、製品長辺全幅から均一充填に近づけるための**中央肉盗み**。**最大5段までネスト可能**で、中央が最薄・最大抵抗、外側に向かって厚みが段階的に増える階段状にできる。

共有パラメータ（`balancer_enabled=True` 時のみ有効）：

| パラメータ | 意味 | 制約 |
|----------|------|------|
| `balancer_enabled` | bool トグル | — |
| `balancer_height_mm` (H_bal) | ▽の頂点〜底辺距離 | apex がバルブゲート円を侵さないこと |
| `balancer_base_distance_from_gate_mm` | 底辺位置の半円中心からの y距離 | `≤ D` |

段ごとのパラメータ — **2通りの指定方式**：

1. **スカラー形（後方互換、1段固定）**:
   - `balancer_base_width_mm` (W_bal) — `≤ W_gate`
   - `balancer_target_thickness_mm` (h_bal) — `> 0`
   - 既存ケース／旧 cfg はこの形のまま動く。

2. **タプル形（1〜5段ネスト、中央→外側の順）**:
   - `balancer_base_widths_mm` — `tuple[float, ...]`（長さ N、`W_1 ≤ W_2 ≤ … ≤ W_N`）
   - `balancer_thicknesses_mm` — `tuple[float, ...]`（長さ N、`h_1 ≤ h_2 ≤ … ≤ h_N`）
   - 段 1 が中央（最薄）、段 N が外側。`W_N ≤ W_gate`、各値 `> 0`。

`resolved_balancer_stages()` で `[(W_k, h_k), ...]` の正規化リストを返す。タプル形が空ならスカラー形を 1段として包む。

**実装**: 外側→内側の順でセルに `h_k` を上書きしていく。結果として、中央軸（cx）からの距離 `|x - cx|` が `≤ 0.5·W_1·t_y` の領域は `h_1`、`0.5·W_1·t_y < … ≤ 0.5·W_2·t_y` は `h_2`、…、最も外側の輪は `h_N` になる（`t_y` は y 方向の補間係数、apex で 0、base で 1）。`h_k = plate_lower_thk` で揃えるとその段のキャビティ天井が**ゲート側プレートの天面と同じ高さ**になる。

**物理**: コンダクタンス `S = h³/(12η)` の `h` を ▽ 内で下げると流路抵抗が `h³` で激増する。中央が最大抵抗、外側に向かって緩やかに減るので、樹脂は中央を避けて長辺の両端側に回り込み、N段ネストにより階段状に流動分配の精密制御が可能。

**設計の勘所**: 段数を増やすほど精密に制御できるが、セルあたりの厚み変化が大きくなり数値条件が悪化する。`h_k` を小さくする・`W_k` を広げる・`H_bal/D` 比を上げると効果が強くなる。`run_demo.py` の `FilmGate_PP_balancer` は 1段（スカラー形）のチューニング起点ケース。

### `build_direct_gate_geometry` の形状仕様

長方形プレート単体 + プレート**内部**にバルブゲート円。ホットランナー系の直接ゲート（バルブゲートピンが製品表面に直接降りてくる）を想定。**ランナーもスプルー帯もない**、樹脂は当該位置から垂直に注入されるという 2D 簡略モデル。

座標系（`y` 上向き、`x` 右向き、Film gate と同じ：ゲート側辺が小 y 側、反ゲート側辺が大 y 側）:

```
y = pad + Hp                ← 反ゲート側辺（far edge）
y = pad + split_h           ← プレート段差線（split_h > 0 のとき）
y = pad + g_off             ← ゲート円中心
y = pad                     ← ゲート側辺（gate-side edge）
```

**パラメータ（`DirectGateConfig`）**:

| 記号 | 意味 | デフォルト | 制約 |
|------|------|------|------|
| `plate_w_mm` (Wp) | 製品幅 | — | > 0 |
| `plate_h_mm` (Hp) | 製品高（ゲート側辺〜反ゲート側辺） | — | > 0 |
| `plate_thk_mm` | 製品肉厚（split=0 で均一、split>0 で `lower_thk_mm` / `upper_thk_mm` の None 時フォールバック） | — | > 0 |
| `gate_diameter_mm` (d) | ゲート円径 | 3.0 | `≤ plate_w_mm` |
| `gate_offset_mm` (g_off) | ゲート側辺からゲート中心までの**内側方向の距離** | 20.0 | `d/2 ≤ g_off ≤ Hp - d/2` |
| `plate_split_height_mm` | ゲート側辺からの段差位置（0 で均一） | 0.0 | `0 ≤ split ≤ Hp` |
| `plate_lower_thk_mm` | ゲート側帯の肉厚（None で `plate_thk_mm` フォールバック） | None | `> 0`（指定時） |
| `plate_upper_thk_mm` | 反ゲート側帯の肉厚（None で `plate_thk_mm` フォールバック） | None | `> 0`（指定時） |
| `cell_size_mm` | メッシュサイズ | 1.0 | > 0 |

**シルエット**: 幅 Wp × 高さ Hp の長方形プレート1枚のみ。ゲート円（半径 d/2）はプレート内部、左右中央線上、ゲート側辺から内側へ g_off mm の位置に存在する。ゲート円のセルもプレートの一部であり、肉厚はそのセルが属する帯の肉厚（バルブピンは製品表面に出入りするだけで、cavity 形状を変えない）。

**肉厚プロファイル**:
- `plate_split_height_mm == 0`（uniform）: 全セル `plate_thk_mm`
- `plate_split_height_mm > 0`: `[gate-side edge, gate-side edge + split_h]` が `plate_lower_thk_mm`、それ以遠が `plate_upper_thk_mm`。Film gate のプレート分割と同じ概念だが、Direct gate はプレート単独なのでスロープゾーンはなく、段差は離散的（隣接2行で肉厚が切り替わる）。

**圧縮マスク**: `compression_mask = in_plate.copy()`。**プレート全体（ゲート円含む）が圧縮対象**。Film gate と異なり、ランナー・スプルーがないので「どの領域が膨張対象か」を区別する必要がない。`compression_volume_fraction()` は 1.0 を返し、圧縮短縮効果は legacy 全セル膨張と同じ式 `T_fill *= compression_fraction/cf + (1 - compression_fraction)` がそのまま使える。

**バリデーション**:
- `gate_offset_mm < d/2` → ゲート円がゲート側辺を突き抜ける、reject
- `gate_offset_mm + d/2 > Hp` → ゲート円が反対端を突き抜ける、reject
- `d > plate_w_mm` → ゲート円がプレート幅を超える、reject
- `plate_split_height_mm` が範囲外、`plate_lower_thk_mm` / `plate_upper_thk_mm` が非正 → reject

### 意図的にモデル化していないもの

- コアのバルク温度低下と粘度の動的更新（スキン層は Stefan/Neumann 近似で扱うが、コア温度は melt のまま固定）
- **面内コーナー効果・ジェッティング・二次流**（層別 Hele-Shaw でも 2D 前提は不変。捕捉には完全 3D FVM/FEM が必要）
- 結晶化・収縮反り
- パッキング段階の保圧
- 中立面メッシュ・非構造格子・STL/STEP 入力
- **剪断発熱の自己整合 (段階2)**: 段階1 (`shear_heating_enabled=True`) は閉形式の局所近似のみ。エネルギー方程式 `ρcp·∂T/∂t = k·∂²T/∂z² + η·γ̇²` を厚み方向 1D FDM で陰解法積分する段階2 は別 PR で対応予定

スキン層モデル (`HeleShawSolver` の `skin_layer_enabled=True`) と層別ソルバー (`MultilayerHeleShawSolver`) は **厚み方向の 3D 性** (壁面凍結、層別温度、層別粘度、局所剪断速度の層別評価) を取り込んでいるので、これらは「モデル化していない」リストから外れる。一方、**面内の 3D 性**（角部の二次流、ジェッティング、コーナー渦）は依然として捕捉できない — Hele-Shaw 系の根本的限界。

これらを「solver に足す」のは前提が崩れる。新機能として別解法を立てて並列に置く方向で考えろ（既存スキン層 / 層別ソルバーが既に並列パターンの実例）。

## テスト

`tests/` 配下に 19 ファイル、合計 **417 テスト** (416 pass + 1 skip — short-shot 高 threshold ケース)：

- `test_smoke.py` — 4件: import / MaterialDB / build_demo_geometry / Cross-WLF 単調性
- `test_solver_1d.py` — 5件: 1Dストリップの解析解 `τ(x) = x(2L−x)/(2S)` との比較。max誤差 <2%、メッシュ細分化で誤差減少を保証
- `test_geometry_film_gate.py` — 43件: シルエット / 厚み / ゲート土手 / 体積スケール / バリデーション / バランサー（1段スカラー形 + N段ネスト） / プレート分割（ゲート側/反ゲート側2層） / solver 統合 / **compression_mask（プレート本体のみ膨張、ランナー・ゲートは不変）**
- `test_geometry_direct_gate.py` — 26件: シルエット（プレート単体・ランナー無し） / ゲート位置（左右中央＋ゲート側辺から `g_off` mm 内側） / ゲート径 / 体積 / 圧縮マスク（プレート全体） / バリデーション（ゲート円の突き抜けチェック含む） / solver 統合 / 圧縮成形による T_fill 短縮 / **プレート分割（ゲート側／反ゲート側2層、resolved_plate_zones、None フォールバック、バリデーション）**
- `test_geometry_film_gate2.py` — 33件: 直角台形シルエット / 厚み場のプロファイル（連続性・段差の有無・x非依存性）/ ゲート位置可変 / 2段テーパ / 深ランナー / `resolved_*` フォールバック / バリデーション / solver 統合
- `test_geometry_profile_gate.py` — 54件: JSON I/O（round-trip、パス付きエラー、未知キー拒否、非オブジェクトのセクション拒否）/ バリデーション / シルエット（ランド・ランプ式・cap 底打ち・アイランド・外壁・対称性 flip・非対称）/ 井戸（フロア深さ・max 合成・体積差分を放射積分と照合）/ **閉形式の体積検算**（直壁最小スペック ±3%）/ グリッドはみ出し拒否 / compression_mask / プレート2層 / solver 統合 / **溶接ダム（`island.weld`）**（帯内は一定深さ・帯外と島外は不変・体積減を求積と照合・未指定時は旧挙動と bit-identical・round-trip・バリデーション）/ **非有限値の拒否**（`validate` の dataclass 走査が数値リーフ28個に届くこと・JSON 経由 × NaN/±Inf・パーサを通さない直接構築・NumPy スカラが走査から落ちないこと）/ **配列フィールドの構造検証**（5フィールド × 壊れた値8種、2文字文字列のアンパック化けを含む）
- `test_spec_source.py` — 44件: **gitignore ルールそのもの**（リンク本体は実リポに問い、配下は `.gitignore` を写した使い捨てリポで判定 — 実リポに問うと「シンボリックリンクの先」で 128 が返り、**機能を実際に使っている環境でだけ skip する**。ルールが無いと落ちることも別途 assert）/ ルールが広がりすぎてデモスペックを飲まないこと（**使い捨てリポの未追跡コピーに問う** — `git check-ignore` は追跡中のファイルにはルールに関わらず「無視しない」を返すので、実リポの実ファイルに問うと `*.json` でも通る空のガードになる）/ デモスペックが実際に追跡されていること / `spec_root`（不在・実ディレクトリ・外を指す symlink・**完全 resolve**・リンク切れ・ファイル・**symlink ループ**）/ `spec_link_exists`（不在と「あるが壊れている」の区別）/ `list_spec_files`（`*.json` のみ・ディレクトリ除外・**mtime 順でなく名前順**・読めないディレクトリで例外）/ `choose_spec_origin` 10 通り + IO を踏まないこと / **配線を AppTest で実 `app.py` に対して**（リンク無しで一覧が出ない・既定が未選択で何も読まれない・選択で読める・**ドロップが一覧に勝ちその旨が出て一覧が disabled になる**・解除で一覧に戻る・リンク切れの告知・読めないフォルダ/ファイルの告知に**パスが含まれない**・記録される名前がファイル名のみ）
- `test_settings_record.py` — 11件: 形状 config の全フィールドが記録されること（dataclass 自身と突き合わせ）/ `None` が null として残ること / tuple が list になり JSON round-trip すること / **アップロードしたスペックの中身がシリアライズ結果に現れないこと**（キーの有無でなく実際の数値で検証）/ フィンガープリントの同一性・差分検出 / bytes と str の等価 / config dataclass を持たない入力（`cfg=None`）/ `spec` 配下のキーが `name`/`sha256`/`bytes` に限られること（スペック自身の `name` フィールドは中身なので載せない）/ 型・空文字の拒否
- `test_version.py` — 5件: `pyproject.toml` と `__version__` の一致 / CHANGELOG に現行版の項目がある / CHANGELOG の版が降順 / `build_label()` が版で始まる / git メタデータ（SHA・日付・dirty フラグ）の反映
- `test_compression_stroke.py` — 9件: stroke モード後方互換（`compression_stroke_mm=None` で factor モードと完全一致）/ 段差プレートで段差保存 / 全 target セル等量加算 / `stroke=0` で圧縮 OFF 一致 / uniform プレートで factor モードと stroke モードが等価 / metadata の `compression_mode` / `compression_stroke_mm` 露出 / `Geometry.compression_area_mm2()` ヘルパー
- `test_short_shot_timeline.py` — 27件: 凍結セルが T_fill を決めないこと / 未充填セルの `fill_time_s` と `pressure_norm` が NaN / 時間軸が `tau_max_flow` 由来であること（`tau_max` との比が 10 倍超）/ **凍結セルに封じ込められた領域も未充填になること**（連結性）/ 凍結が無ければ挙動不変 / スキン OFF で `unfillable_mask` は None / ゲート凍結で全キャビティ未充填 / `_tau_reference` のフォールバック / metadata の内訳 / **色の場が死んだセルを隣の生きたセルから取ること**（合成データで検証。実部品では隣が最遅セルになりがちで、実物ベースだと差が出ない）/ 死んだセルが `filled` に関係なく覆われ続けること / タイトルのショートショット表記 / **未充填セルにウェルド・エアトラップを立てないこと**（生 τ なら立つことを前提 assert してから検証）/ 有限な充填時間が1種類でもウェルド図が描けること / 圧力マップが未充填セルを専用色で塗ること（アルファ固定後の画素で検証）/ **ゲート以外が全部封じ込められたとき T_fill がベースラインに留まること**（全体最大へのフォールバックは死セルの τ を呼び戻す）/ 描画の時間軸とヘッドラインが一致すること / **色スケール・タイトル・フレーム時刻の3者が同じ軸を読むこと**（欠陥は重複そのものなので3つまとめて assert）/ 反復途中で全セルが凍っても落ちないこと（反復上限3/5/8）/ **生きた領域を切り離して解き直していること**（独立に組んだ live-only 解と 25% 以内、未再解なら 3.3 倍ずれる）/ ベースライン時間が充填する体積だけを数えること / inflation が同じ領域の2状態の比であること / **射出率未指定の既定経路でも体積スケールが効くこと**（他のテストは全部明示的に率を渡していて、この経路を通らなかった）
- `test_skin_layer.py` — 6件: skin OFF/ON、`c_skin=0` で baseline 復元、極薄肉での short shot 検出、metadata の整合性
- `test_multilayer_solver.py` — 42件: 層分布プリミティブ (uniform / wall_refined / 端点・対称性・壁細密性・plan 例一致・Σm=1/6) / コンダクタンス helper (N=1 で h³/12η、cavity 外ゼロ、(N,) と (N,ny,nx) η 形状) / N=1 で既存 `HeleShawSolver` と一致 (anchor) / Σh_k=h_total / 後方互換 / wall_refined ソルバー受理 / 温度結合 (layer フィールド populated/None、τ_max 変化、収束性、tol 感度、metadata、壁<中央温度) / ショートショット (metadata 存在、warm で 0、極薄+高 threshold で発火、threshold 0 で 0) / damping (metadata、引数検証、ω=1 動作) / **剪断発熱段階1** (既定 OFF で後方互換、Br 数は常に populated、ON で ΔT_max>0 + 層フィールド shape、ON で η が下がる、material 由来 cp/k メタデータ確認)
- `test_multilayer_thermal.py` — 22件: Neumann 1D (t→0 で T_melt、t→∞ で T_mold clamp、対称性、中央 > 壁、t 単調性、入力検証) / Poiseuille (壁で max、中央 floor、shape、floor=0、引数検証) / **剪断発熱段階1** (shape & 非負、γ̇=0 で ΔT=0、γ̇² スケーリング、t≫τ_thermal で頭打ち、極薄 PP の桁感、shape 不整合検出) / **Brinkman 数** (shape & 非負、γ̇=0 でゼロ、極薄高速で Br>1、k と ΔT の非正検出)
- `test_visualizer_3d.py` — 25件: block anatomy（**3トレースとも flat-top Mesh3d**、天面=各 cavity セル2三角形＝面数 2×cavity・`intensitymode="cell"`）、天面が全 cavity セルを覆う（面数=2×cavity＝侵食ゼロ・天面 Z 有限正・床 Z=0）、**flat-top が対角境界を塞ぐ**（対角バンドで面数=2×cavity）、頂点がゲート中心座標（セル端の隅）、境界壁が PL〜天面を覆う、`aspectmode='data'` で等倍、天面=面ごと/側壁=頂点ごと intensity で coloraxis 共有、**厚み段差が縦の段差として描かれる**（段差プレートで天面 z が両厚みを保持＋PL非接触の段差壁が立つ）、**天面ホバーが場の値を露出**（fill/pressure で per-vertex customdata が読める）、**肉厚配色のガード**（2D の `THICKNESS_CMAP` と一致すること／ramp 全域で明→暗に単調であること／薄肉端と厚肉端のコントラスト比が 3:1 以上あること／colorscale の stop が hex と `rgb(...)` のどちらでも読めること）、**基準そのものの判別力**（`Cividis_r` / `Blues` / `Blues_r` / `Jet_r` / `Bluered_r` / `Greys_r` で期待値を固定。定数を差し替える変異注入は名前一致 assert が先に落ちて判定にならないため、基準を直接テスト）、**stop 判定だけでは区間内の逆行を見逃すこと**（緑→マゼンタの**構成した**反例で固定。組込配色を証人にすると、その配色が変わったときに「サンプリングはもう不要」と誤読される）。各ガードを何故そう組んだかは上の `visualizer.py` 項を参照
- `test_fill_render.py` — 24件: 既定配色が turbo であること / 色スケールが 0..T_fill 固定 / 色の層が完全不透明（アルファに mask を持たせない）/ キャビティ外に NaN が残らない / `_nearest_extend` の3挙動 / **オーバーレイが露出するセルが `mask & filled` と完全一致**（侵食も滲みもゼロ）/ 外側とキャビティ内未充填が別の灰色 / 等時線のレベルが T_fill 固定でフレーム間不変 / 等時線 OFF と空フレーム / 全配色でのレンダリング / フレーム PNG が自前のカラーバーを持つこと / **等時線が一度だけ描かれ zorder でオーバーレイの下に入ること**（呼び出し順ではなく zorder で検証。matplotlib は imshow=0 / contour=2 なので、後から呼ぶだけでは隠れない）/ 要求した本数ちょうどが描かれること / 1セル幅キャビティで等時線を諦めること / ゲートマーカーがオーバーレイより上に来ること / **60枚出力しても contour が1回だけであること**（figure を毎フレーム作り直すと出力は同一なので、contour 呼び出し回数を数える以外に検出手段がない）
- `test_fill_player.py` — 18件: フレーム時刻の単一ソース契約（GIF・PNG 連番・プレイヤーの三者一致）/ 充填率の単調性と末尾 1.0 / payload の埋め込みとデコード / オフライン自己完結（`http(s)://` を含まない）/ 入力検証 / ネイティブ幅キャップと component 高さの導出 / **単体 HTML 化**（`<meta charset>` が最初の非 ASCII バイトより前、文書完全性、フラグメント同一性、title/note のエスケープ）
- `test_visualizer_layer.py` — 15件 (1 skip): `render_layer_map` 4 field smoke / 不正 field / 範囲外 layer_idx / thermal_off で field 別動作 / `render_layer_grid` / `render_short_shot_map` (flagged あり/なし、後者は skip 想定可) / `_scalar_layer_field` helper / ζ レンジが metadata に乗ること / **`THICKNESS_CMAP` が明→暗で走ること**（薄肉が明るい側）と**薄肉端が彩度を保つこと**（HSV 彩度 > 0.5。白に寄る単色マップを弾く）/ 層別 thickness パネルが同じ ramp を使うこと

新機能を足したら**該当する系統のテストファイルにテストを追加**するのが慣例。形状なら `test_geometry_*.py`、solver の挙動なら `test_solver_*.py` か `test_skin_layer.py` か `test_multilayer_solver.py`、純関数の helper なら `test_multilayer_thermal.py`、3D 系なら `test_visualizer_3d.py`、層別可視化なら `test_visualizer_layer.py`。

## バージョン運用

`core/version.py` の `__version__` が**版の単一ソース**。UI サイドバー最下部に
`build_label()` の結果（例 `v0.14.0 (ad8da46, 2026-08-07)`）が出る。**コミット SHA を
併記する目的は、デプロイ済みインスタンスが最新かを判別すること**（Streamlit Cloud は
Reboot するまで古いビルドを配ることがある）。作業ツリーが dirty なら `+dirty` が付く。

版を上げるとき（機能の節目ごと、`0.x` 系なのでマイナーを刻む）:

1. `core/version.py` の `__version__` を更新
2. `pyproject.toml` の `version` を同じ値に更新
3. `CHANGELOG.md` の先頭に `## [x.y.z] — YYYY-MM-DD` の節を追加
   （冒頭に「何ができるようになったか」を太字1行 → 追加 / 変更 / 削除。**撤回した機能や
   既知のトレードオフも残す**）
4. マージ後に `git tag -a vx.y.z` を打つ

1〜3 の整合は `tests/test_version.py` が検証するので、**上げ忘れると CI が落ちる**。
版表示はサイドバー内に置くこと（メインフローには `st.stop()` が多数あり、その後ろに
置くとパラメータ不整合時に版が消える）。

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
- `pytest tests/` （56件全部 pass を確認）

CI 設定: `.github/workflows/ci.yml`。Python 3.11 / 3.12 マトリクスで上記3つを走らせる。Node.js 20 actions の deprecation warning が出るが2026-06-02までは無害。

## データ・依存

- `data/materials.json` — Cross-WLF パラメータ8樹脂（PP, PP_T10, PP_T20, PP_T30, ABS, PC, PA66, PMMA）。**generic 値**であり、実プロジェクト用にはベンダー実測データを差し替える前提。PP_T10 / PP_T20 / PP_T30 は PP 純品からの経験則ベース補正（D1 ×1.15/1.40/1.80、α と密度はタルク重量分率の調合計算）。UI の樹脂セレクタの初期選択は PP_T20。
- `.codex` — 0バイトの外部ツールマーカー（gitignore 済み）。
- **依存の二重管理**:
  - `pyproject.toml` の `[project] dependencies` がローカル開発の正本（`uv pip install -e ".[dev]"` で読まれる）。
  - `requirements.txt` は **Streamlit Community Cloud デプロイ用**のミラー。pyproject.toml の deps を変更したら必ずこちら側も同期する。
  - `runtime.txt` は Streamlit Cloud に Python バージョンを伝える1行（`python-3.12`）。
  - `.streamlit/config.toml` は Streamlit ランタイム設定（アップロード上限等）。ローカル/Cloud 両方で読まれる。
- 主な依存：numpy / scipy（ソルバ）、matplotlib（画像書き出し）、Pillow（GIF 書き出しとプレイヤーのフレーム寸法取得）、streamlit（UI）、**plotly**（3D表示、`visualizer_3d.py` 専用）。**注意**：Streamlit は畳んだ expander の中身も毎 rerun で実行するので、3D図3枚は expander が畳まれていても毎回構築・送出される（「開いた時だけコストが走る」わけではない）。3D は疎 Mesh3d でネイティブ解像度描画する（点数は cavity セル数ぶん）。3D の回転負荷を下げるためフルグリッド Surface を使わない設計。

## UI と CLI の対応関係

`app.py` のサイドバー入力は4系統（`Film gate 1` / `Film gate 2` / `Direct gate` / `Profile gate`）。**画像入力（PNG/JPG 二値化）は v0.23.0 で撤去した** — 一度も使われず、PDF から JSON スペックを起こす経路が確立して役目が無くなったため。`core.geometry_from_image` ごと消したので、戻すなら git 履歴から拾う。`run_demo.py` の `DEMO_CASES` / `FILM_GATE_CASES` / `DIRECT_GATE_CASES` も同じパラメータ群を扱う（`DEMO_CASES` は CLI 専用、UI からは消した）。**新パラメータを `HeleShawSolver` / `FilmGateConfig` / `DirectGateConfig` に足すなら、UI と CLI の両方に反映する必要がある**。

特に `FilmGateConfig` の `D_flat + D_slope = D` 制約は、UI 側では「`D_flat / D` の比率スライダー」で表現してこの制約を自動満足させている（`app.py` の `flat_ratio` 変数）。CLI 側は直接 `D_flat` / `D_slope` を渡すので、case 定義時に和が `D` になることを手動で保証する必要がある。

その他の UI ↔ CLI ブリッジ方針：

- **バランサー**: UI は段数 N をスライダーで選び、`balancer_base_widths_mm` / `balancer_thicknesses_mm` のタプルを構築して渡す。スカラー形（1段固定）は CLI の旧ケース互換のため温存。CLI で N段にするなら直接タプルを書く。
- **プレート分割**: UI は「段差位置 [mm]」スライダーで `plate_split_height_mm` を出し、値が `0` のときはゲート側／反ゲート側の肉厚スライダーを隠して `plate_thk` 1本に統合、cfg には `plate_lower_thk_mm = plate_upper_thk_mm = None` を渡して uniform モードに落とす。CLI で uniform にしたいときも同じく `plate_split_height_mm=0` ＋ `plate_lower_thk_mm = plate_upper_thk_mm = None` で足りる。
- **スキン層**: UI のトグルが `skin_layer_enabled`、スライダーが `skin_growth_constant`（`c_skin`）。CLI 側はソルバー kwargs に直接渡す（`run_demo.py` の `PP_skin_layer` 参照）。
- **射出条件の単位系**: `injection_velocity_mms` / `injection_volume_flow_cm3s` の既定値は **実機ユニット域**（`eae5394` で再スケール済）。新ケースを足すときも実機相当の値を入れる前提で考える。CLI 既定値（`run_demo.py`）と UI 既定値（`app.py`）はこの方針で揃えてある。
- **Direct gate**: UI では「製品幅 / 製品高 / 段差位置 / ゲート側肉厚 / 反ゲート側肉厚 / ゲート径 / ゲート位置」のスライダー群で `DirectGateConfig` を組み立てる。プレート分割の挙動は Film gate と同じ（段差位置 0 で uniform、`> 0` で 2 層化、cfg には `plate_lower_thk_mm` / `plate_upper_thk_mm` を渡す）。デフォルト値は Film gate と揃えてある（Wp=300 / Hp=50 / 段差=20 / lower=0.35 / upper=0.50 / Φ=3 / g_off=20）。ゲート位置スライダーの上下限はゲート径とプレート高さに連動して動的に計算（突き抜けバリデーションを UI 側でも防御）。CLI でも `DirectGateConfig` の引数に同じ制約がある。
- **圧縮成形のスコープ**: 圧縮で膨らむのは `Geometry.compression_mask` が True のセルだけ。Film gate のビルダーは「プレート本体だけ True」（ランナー・スプルー・ゲートは膨張しない）、Direct gate のビルダーは「プレート全体 True」（cavity = プレート単体なので全部膨張）。`build_demo_geometry` は `compression_mask=None`（旧挙動＝全セル膨張）。新しい形状ビルダーを足すときは「製品本体だけ True」の compression_mask をセットすること。
- **圧縮量の指定方式**: UI は ICM ON 時にラジオで `factor` / `stroke` を選ぶ。factor 選択時は「初期隙間倍率 h_init/h_final」スライダー（`compression_stroke_mm=None` で solver に渡る）、stroke 選択時は「圧縮ストローク [mm]」スライダー（`compression_factor=1.0` ＋ `compression_stroke_mm=<値>` で渡る）。CLI 側は両方を `make_solver()` の kwargs に直接渡し、ケース定義で片方だけ指定する（factor モードならデフォルト、stroke モードなら `compression_stroke_mm=0.70` 等を明示）。段差プレート（plate_lower_thk ≠ plate_upper_thk）の圧縮シミュレーションでは **stroke モード一択**（factor モードだと段差が崩れる）。`FilmGate_PP_stepped_stroke` がこの想定の CLI 参照ケース。
- **壁面冷却モデル**: UI ヘッダー「壁面冷却モデル」のラジオで `なし` / `スキン層` / `層別` の 3 択。**排他選択**により skin と multilayer の同時 ON は構造的に不可能。極薄プレート (t<0.5mm 想定) 向けに UI 既定は **層別 / N=7 / wall_refined / max_iter=12**。
  - **なし**: 既存 `HeleShawSolver`、温度結合なし。
  - **スキン層**: `HeleShawSolver(skin_layer_enabled=True, skin_growth_constant=c_skin, ...)`。Stefan/Neumann 1 層モデル。
  - **層別**: `MultilayerHeleShawSolver(num_layers=N, layer_distribution="wall_refined", thermal_coupling=True, ...)`。Cross-WLF 結合 N 層モデル。スライダーで `num_layers` (3..9、既定 7) / `layer_distribution` (`wall_refined` 既定 / `uniform`) / `max_iterations` (1..20、既定 12) / `convergence_tol` / `solidification_temperature_fraction` を出す。**剪断発熱補正 (段階1)** はチェックボックス `shear_heating_enabled` (極薄向け既定 ON)。
  - CLI 側は `_solve_and_export(multilayer=True, num_layers=..., layer_distribution=..., shear_heating_enabled=..., ...)` で明示。`skin_layer=True` と `multilayer=True` の同時指定は `ValueError`。`FilmGate_PP_multilayer_5L` が層別の参照ケース、`FilmGate_PP_multilayer_5L_shear` が剪断発熱 ON の比較ケース (高 V、N=7、極薄)。
  - 結果ペインに「層別プロファイル (Multi-layer N=...)」expander が現れ、温度グリッド / 粘度グリッド / ショートショットマップを表示、各 PNG ダウンロード + ZIP exports に同梱。**剪断発熱メタデータ** (ΔT_max / ΔT_mean / Brinkman 数 max & mean、信号灯 🟢/🟡/🔴) は expander 直下のキャプションに出る。
- **ゲートスペックの読込 (`core/spec_source.py`)**: 「スペック入力」ラジオは `デモプリセット / ローカルから読込 / JSON貼り付け`。分岐は `SpecMode` enum で行う（旧実装はラベルの `startswith` を見ていたので、文言を直すと挙動が変わった）。**ローカル一覧が出るのは、リポ直下の `local_specs` がディレクトリに解決するときだけ**。これは gitignore 済みの名前に**固定**してあり、設定で変えられない。理由は2つ：(1) 環境変数だと**フェイルオープン**で、Streamlit Cloud は secrets のトップレベル `str`/`int`/`float` を `os.environ` に昇格させるため、ローカルの `secrets.toml` を設定欄に貼るだけで公開インスタンスにファイル読み取り口が生える。gitignore されたパスは checkout + secrets で組まれる Cloud 上に存在させる手段が無い。(2) root を設定可能にすると、リポ内を指して `git add -A` するのが**顧客寸法を public リポに commit する最短経路**になり、env ゲートもパス閉じ込めもこれには効かない。固定すれば構造的に起きない。**テキストのパス欄は無い** — ユーザー入力のパスが存在しないので `..`／`~` 展開／prefix 一致／拡張子縛りの問題が書き忘れではなく発生しない。一覧の既定は `— 未選択 —` センチネル：`build_geometry()` は実行ボタンの後ろではなく**毎 rerun 走る**ので、既定で実ファイルを選ぶとページを開いた瞬間に顧客形状が描画される。ドロップと一覧が同時に生きているときは**ドロップが勝ち**、勝った側をサイドバーに明示して負けた一覧を `disabled` にする（帯はメインカラムでなくサイドバーに出す — 食い違っているウィジェットの隣でなければ届かない）。**例外の文言を画面に出すな**：絶対パスが顧客名と案件名を含み、`client.showErrorDetails` の既定は `full`。`Path.resolve()` は symlink ループで `OSError` でなく `RuntimeError` を投げ、`Path.glob()` は権限エラーを飲み込んで空を返す（このため一覧は `os.scandir`）。CLI 側にこの読込経路は無い（`run_demo.py` はケース定義に直接スペックを書く）。
- **結果 ZIP の中身**: 「⬇ GIF + フレーム画像をダウンロード」は `fill.gif` / 各マップ PNG / `frames/` の連番 PNG / `metadata.json`（解析結果）/ **`settings.json`（入力設定）** に加えて **`player.html`** を含む。`settings.json` は形状 config の全フィールド・材料・射出条件・壁面冷却・圧縮・出力・版を持つ（`core/settings_record.py`）。**アップロードしたスペック JSON は名前と SHA-256 だけを記録し、中身は載せない** — ZIP は人に渡す前提なので、実図面由来の寸法を同梱しない。UI に埋め込んでいるのと同じプレイヤーを `wrap_standalone_html()` で完全な HTML 文書に包んだもので、ダブルクリックすれば追加ソフト無しでコマ送りできる。**フラグメントをそのまま `.html` として出すな** — `<meta charset="utf-8">` が無いと `file://` で開いたブラウザが CP932 等にフォールバックして日本語ラベルが化ける（`tests/test_fill_player.py` が charset の位置を検証している）。同梱で ZIP はおよそ 2 倍になる（base64 は 4/3 に膨らみ、PNG は圧縮済みで deflate が効かない）。
- **剪断発熱補正 (viscous dissipation, 段階1)**: 層別モード専用。`ΔT_shear,k = (η_k·γ̇_k²)·min(t_arr, τ_thermal)/(ρ·cp)`、`τ_thermal = h²/(π²·α)`。fixed-point ループで前イテレーションの `η_k` から ΔT_shear を計算 → Neumann 温度に加算 → Cross-WLF で η 再評価。負のフィードバック (T↑ → η↓ → 発熱↓) なので発散しにくい。**Brinkman 数 `Br = η·γ̇²·h²/(k·ΔT_ref)` は補正 OFF でも常に計算**してメタデータに出すので、必要性を事前判定できる。Br>2 は段階2 (1D FDM 陰解法) が本来必要なシグナル。材料 DB 拡張: `specific_heat_J_kgK` 追加 (8 樹脂)、熱伝導率は `k = α·ρ·cp` で派生。
