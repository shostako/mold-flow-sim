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
- **`solver.py`** — `HeleShawSolver` と結果 `FlowResult`。中核アルゴリズムは下記。**充填しないセルは充填時間を持たない**: スキン層が閉じたセル（`min_core_thickness_mm` で床打ちされる）と、それに封じ込められてゲートから到達できなくなった領域は `FlowResult.unfillable_mask` に入り、`fill_time_s` / `pressure_norm` は NaN になる。**未充填領域を切り離してスキン層の不動点ごと解き直し**（`_solve_domain` を live 領域で回し、新たに凍結が出ればもう一度切り出す。`MAX_DOMAIN_PASSES=4` は暴走止め。cavity 全体で解くと死んだ領域の体積が上流の生きたセルを通り、その τ が膨らむ — 実測3.3倍）、絶対時間の基準はその解の τ（`tau_max_flow`）から取る。ベースライン時間も充填する体積で計算する — 閉じたセルの τ は桁違いに大きく、基準に入れるとショートショットが「時間がかかる」に化けて T_fill が数倍に膨らむ。
- **`multilayer_solver.py`** — `MultilayerHeleShawSolver` と継承結果 `MultilayerFlowResult`。厚み方向を `N` 層に離散化して **層別温度・粘度** の coupling を fixed-point で解く新ソルバー。既存 `HeleShawSolver` を内部に保持して helper メソッド (`_effective_viscosity` / `_open_thickness_field` / `_solve_tau_field` / weld・airtrap) を委譲で再利用、線形代数の重複なし。詳細は後述「層別 Hele-Shaw ソルバー」セクション。
- **`multilayer_thermal.py`** — 純関数 `neumann_layer_temperatures()` と `poiseuille_shear_rates()`。前者は両壁から育つ Neumann 半無限解の重ね合わせ `T(z, t) = T_mold + (T_melt - T_mold) · [erf(z/(2√(αt))) + erf((h-z)/(2√(αt))) - 1]` (長時間極限で `T_mold` 以下に落ちるので clamp)、後者は Poiseuille 解析微分 `γ̇_k = (6V/h) · |2ζ - 1|` (中央発散回避の floor 付き)。両方とも shape `(N, ny, nx)` を返す。
- **`two_phase.py`** — 二相ショートショットモデル `solve_two_phase_short_shot(solver, shot_volume_cm3)` と結果 `TwoPhaseShortShotResult`。計量制限（意図的ステージング）のショートショットを実機パラメータのまま予測する。(1) **射出相**: 型開きギャップ（`_open_thickness_field()`、compression 設定を継承）で τ を解き、体積CDFを計量体積で切って溶融プール Ω₁ を得る。(2) **圧縮相**: 最終肉厚で τ を再解 — **Ω₁ 全セルを Dirichlet（等圧ソース近似）**に置き、最終肉厚での体積が V_shot になるまで τ 順に前進（体積保存）→ Ω₂。線形求解2回、時間積分なし。同値 τ のタイ群は**原子的**（群全体が入るときだけ取る = `_arrival_time_field` の群末尾時刻と同じ意味論）。**ゲート群（τ=0 のタイ群）を覆えない計量は入口で `ValueError`**（強制包含すると計量以上の達成体積を報告する — Codex P2 対応）。**位相2はギャップが実際に閉じるときだけ走る**（`gap_closes` ガード。ICM OFF / stroke 0 では原子的カットの残余 budget があっても τ₂ を解かず Ω₂ = Ω₁ — 非一様肉厚で残余が小セルに化ける穴を封じる、同 P2 対応）。**スキン層モデルとの併用は入口で `ValueError`** — 計量律速の短絡は凍結を含まないという用途定義。制約: 圧縮中の凍結なし・射出/圧縮オーバーラップなし・プール内圧損無視（薄板で妥当）。**開ギャップを縮める圧縮設定（負 stroke / factor<1）は入口で `ValueError`**。**ネスト単調性は Ω₁ のみ**（Ω₂ は計量ごとに τ₂ の Dirichlet 境界が違うので保証外 — 計量スイープは「入れ子のフィルム」でなく独立な準静的履歴の族）。**V_fin ≤ V_shot < V_open（ICM 通常運転の完全充填）でも τ₂ を解いて圧縮順序（`compression_progress`）を報告する** — 形状は自明でも履歴は自明でない
- **`visualizer.py`** — `render_fill_animation`（GIF）、`render_pressure_map`、`render_weldlines`、`render_skin_layer_map` / `render_core_layer_map`（スキン層 ON 時のみ意味あり）、`render_layer_map` / `render_layer_grid` / `render_short_shot_map`（層別ソルバーの結果用、後述）、`render_two_phase_map` / `render_two_phase_animation`（二相ショートショットのカテゴリカルマップと履歴 GIF: 青=射出で充填＋射出等時線、橙=圧縮で前進、灰=未充填。順序なしの事実なので ramp ではなく不透明カテゴリ色 — 領域境界そのものが成果物。**凡例は figure レベルで図の下** — 横長プレートでは axes 内のどのコーナーも製品に被る。レイアウト確定（tight_layout）は**タイトル設定後** — 先に呼ぶとタイトルが上端で切れる。アニメのフレーム系列は `core/two_phase.py` の純データ `frame_states()` が供給: 射出相は実時間・圧縮相は正規化順序、フレーム配分は各相の充填セル数比で最低3枚ずつ）、`export_frames`（PNG連番）。matplotlib で `Agg` バックエンド固定。**画像書き出し系**（PNG/GIF）はここ。**充填アニメの配色は `FILL_CMAP = "turbo"`**（`viridis` から変更）。理由は3つで、(1) 成形屋が読むのは絶対値でなく**等時線の形**（詰まる=遅い / ぶつかる=ウェルド / 途切れた先=最後）で、虹の色相コントラストがその縞を浮かせる（明度単調な viridis は逆に均す）、(2) **赤=最後に充填=リスク箇所**という意味論が一致する（viridis は上端が明るい黄色で逆）、(3) 商用 CAE と同じ配色なので相手に読み方の説明が要らない。`jet` でなく `turbo` なのは、jet がシアン/黄で明度が跳ねて**データに無い偽の縞**を描き、等時線と紛れるため。UI で turbo / jet / viridis / cividis を選べる（赤緑色覚への配慮は選択で担保）。**`_draw_isochrones`** が等時線を重ね、**`_fill_field_rgb` + `_unfilled_overlay` の2層描画**で「色は滑らか・輪郭はセル厳密」を両立する — 色の層は `_nearest_extend` でキャビティ外まで延長した不透明 RGBA を `bilinear` で描き、その上から未充填セルを `nearest` の不透明オーバーレイで塗り潰す。**充填時間は連続場なので色の補間は嘘にならないが、輪郭と溶融先端は mask のとおりが正しい**ので、アルファに mask を持たせて補間する（＝先端が半セル滲む）ことは避けている。**肉厚場は本質的に不連続**（t0.35 と t0.50 の間に中間肉厚は実在しない）なので、この平滑化を肉厚マップへ横展開してはいけない — 3D の flat-top 判断と同じ原則。**肉厚マップの配色は `THICKNESS_CMAP = "cividis_r"`**（`cividis` から反転）。肉厚は解いた結果ではなく**入力＝設計者が描いた形状**なので、結果図の虹ではなく単一量の明度 ramp を当てて一目で別カテゴリに見せる。向きは**明＝薄肉 / 暗＝厚肉**で、「インクの濃さ＝物質の量」という普遍的な直観に合うほか、透明樹脂の成形品では**厚いほど光が減衰して実際に暗く見える**（LGP を見慣れた目には反転前が現実と逆）。単色系（`Blues` / `bone_r`）でなく `cividis_r` なのは、**薄肉端が彩度を保つ必要がある**ため — 製品プレートは最薄領域であると同時に唯一注視される領域で、低端が白に寄るマップは製品を白飛びさせ、隣接ゾーン間の段差コントラストを潰し、3D では天面が薄グレーの PL 床と白背景に溶ける（`bone_r` で実測、0.35mm 帯が純白になり外形が消えた）。加えて cividis は色覚多様性向けに設計されており、その性質は反転しても残る。

  **配色ガードの組み方**（`tests/colorimetry.py` に色の計算を集約、`tests/test_visualizer_3d.py` / `test_visualizer_layer.py` が使う）: 配色は名前で照合せず**実際に描かれる ramp を測色して**判定する。名前だけの比較は `Cividis_r` が実は反転していなくても通る。(1) **単調性** — ramp を stop 間まで密サンプルして隣接サンプルが必ず暗くなること。両端だけの判定は少なからぬ組込配色に破られる（plotly 6.7.0 実測で 188ramp 中 26本。`plotly>=5.18` は上限無しなので件数は不変条件ではない）。最悪の `jet_r` は両端の輝度差 0.070 に対し中間で 0.719 逆行する＝「明→暗」を名乗るレインボー。密サンプルが要るのは Plotly が符号化 sRGB で線形補間する一方 relative luminance が非線形変換を先に掛けるためで、伝達関数が凸なので、区間内の輝度は暗い側の端点より下に沈んでから戻りうる — **これは色空間の性質であって特定の配色の性質ではない**。`bluered_r` が今それを示している（0.0023）が、証人は構成した反例（緑→マゼンタ）で固定してある。(2) **可読性** — 薄肉端と厚肉端のコントラスト比 ≥ 3:1（WCAG 1.4.11）。単調なだけなら近黒→黒の ramp も通り、順序は復元できても地図が読めない。(3) **白との分離** — 薄肉端の HSV 彩度 > 0.5。`cividis_r` の黄は白に対し輝度では 1.25:1 しかなく、**白背景から分離しているのは彩度であって明度ではない**ので、明度で判定すると採用した配色自身を弾く。輝度は必ず sRGB を線形化してから求める — ガンマ符号化のまま加重和を取るのは luma であって relative luminance ではなく、`rgb(205,78,46)→rgb(242,34,36)` のように符号が食い違う。
- **`visualizer_3d.py`** — `render_3d_thickness_map` / `render_3d_fill_time` / `render_3d_pressure`。**インタラクティブな3D表示**用。Plotly の `go.Figure` を返し、Streamlit の `st.plotly_chart` で埋め込む想定。各図は **PL（Z=0）= 半透明の薄グレー床 + 側壁 + 天面（Z=h、物理量で着色）** の3トレース構成で、**3トレースとも cavity セルだけを張る疎な `go.Mesh3d`**（天面・側壁は `coloraxis="coloraxis"` 共有、1本のカラーバー）。**各 cavity セルを「厚み一定の平らな天面ブロック」として描く**（flat-top）：`_cavity_corner_mesh` がセル端（隅）を頂点にした flat-top quad を組み、**同一厚みの隣接セルは隅を共有**して頂点を約1×（cavity セル数相当）に抑える（厚みが違うセルは隅を分けて段差の隙間を作る／`np.unique` でベクトル化、Python per-cell ループ無し）。天面の着色は**面ごと**（`intensitymode="cell"`、各セル＝平らな1パッチ＝2D マップと一致）。**厚みの段差はこの flat-top＋段差壁で「くっきりした縦の段差」として描かれる**（セル中心補間だと段差が1セル幅の傾斜に均されてしまうのを回避）——`_build_side_walls` が境界壁（cavity 端、PL→厚み）に加えて、隣接セルの厚みが変わる内部辺に**段差壁**（2つの厚みを繋ぐ縦 quad、+x/+y 方向のみで各辺1回・owner=高い側）を立てる。全 cavity セルが個別 quad を持つので**境界の侵食はゼロ**（旧セル中心三角形化が塞げなかった3セル角・1セル幅も完全カバー）。**【意図したトレードオフ】** 厚み差を全部「段差」扱いするので、**意図的に連続なテーパ**（`build_film_gate_geometry` のランナースロープ等＝多セルで補間）は滑らかなランプでなく微細な階段に描かれる。設計段差（1セルで Δ≈0.15）と連続勾配（≈0.15/セル）は**1セル差が同値で magnitude では区別不能**、唯一の堅牢な信号（局所曲率/二階差分）はランプ端で脆く表示専用機能には過剰なため、製品面の段差（プレート分割・バランサー）をくっきり出すことを優先して flat-top を選択（階段化するテーパは流路側＝離散化データそのままの正直な描画）。Codex P2 を承知の上で維持。**天面・床は元々フルグリッドの `go.Surface` だったが**、Surface は cavity 外も NaN で全格子を持つうえグリッド／ライティング機構を抱える重トレースで WebGL 回転が重く、疎 Mesh3d 化で回転負荷を大幅減。肉厚図の配色は `THICKNESS_COLORSCALE = "Cividis_r"`＝2D の `THICKNESS_CMAP` と同じ ramp で、設計図と立体図が「厚い」の見た目で食い違わないようにしてある（Plotly は名前を大文字化するので定数は別持ち、整合は `tests/test_visualizer_3d.py` が担保）。**`aspectmode="data"`** で x/y/z すべて mm 等倍（誇張なし）。プレートが薄板に見えるのは実物比率そのもの。物理は 2D Hele-Shaw のまま、表現上の3D化のみ。**3D は常にネイティブ解像度**（解析メッシュそのまま）で描画する。かつて「表示専用の解析的精細化」（外形を `cell_size/k` で再ラスタ化して 3D だけ細かくする `refine_for_display` 機能, PR #41）を入れたが、**疎 Mesh3d 化しても精細化時の描画が重く、効果に見合わないとして撤回**（精細化機構は削除、Mesh3d 化は維持）。3D を細かくしたいときは**解析メッシュ自体を下げる**（解析が重くなる代わりに 2D 結果も 3D も細かくなる）運用。

### 中核アルゴリズム（`solver.py`）

充填過程を**時間ステップで進めない**。代わりに楕円型問題を一発解する：

```
-∇·(S ∇τ) = 1   in cavity      ← 実装はこの符号
τ = 0           at gates (Dirichlet)
S∇τ·n = 0       at walls (Neumann, no-flux)
S = h³ / (12·η_eff)   ← Hele-Shaw コンダクタンス
```

符号は `solver.py` の docstring と `README.md` も**この形に揃えてある**（v0.24.0 で統一）。以前は両方が `∇·(S∇τ) = 1` と書いていて実装と食い違っていた。結果 τ は正しかったが、読む側が符号を追うたびに引っかかる。連続形で書くなら `∇·(S∇τ) = -1`。

**`A` は SPD ではない。** この符号で制約前の作用素は対称半正定値だが、`_build_linear_system` は Dirichlet を**行にしか**適用しない（ゲート行を単位行に潰し、隣接する内部行はゲート列の `-coeff` を残す）ので、組み上がった行列は非対称。CG / AMG に載せ替えるならゲート列の消去が先。消去は厳密（ゲートで `τ = 0`）だが、`spsolve` が対称性を要求しないので現状は未実施。

- `_build_linear_system` で5点ステンシル CSR を組み、`scipy.sparse.linalg.spsolve` で `τ` を解く。面コンダクタンスは隣接セルの**調和平均**。**注意：行列組立がPython二重ループでN大に弱い**。中規模以上のメッシュでは性能ネックになる（vectorization 候補）。
- `τ` は擬似到達時間場。絶対時間化は**体積CDF写像**（v0.25.0, Issue #52）: セルの充填時刻 = 「そのセル以下の τ を持つセル群の体積」/ Q。定率射出なら先端は体積線形で進むので、これが物理そのもの（旧 `(τ/τ_max)·T_fill` 線形写像は健全な1Dストリップでも中央を 0.75T と誤報していた — 正しくは 0.5T）。外れ値1セルは自分の体積分しか他セルの時刻を動かせない。同値 τ はグループ末尾の時刻を共有。
- `η_eff` はバルク温度 `0.7·T_melt + 0.3·T_mold` と代表剪断速度で Cross-WLF を1回評価する**定数値**（局所反復なし）。
- 圧縮成形 (`compression_molding=True`) は時間ステッピングではなく、`h` を膨らませて `T_fill` を `compression_fraction/effective_factor + (1-compression_fraction)` で短縮する**等価モデル**。膨張対象は `compression_mask & mask` のセル（None なら全 cavity）。**2 モード対応**：
  - **factor モード**（デフォルト、後方互換）: `h_eff = h * compression_factor`。`effective_factor = 1 + (compression_factor − 1) · f_comp`、`f_comp = Geometry.compression_volume_fraction()`（圧縮対象セル**体積** / 全 cavity 体積）。同じ倍率を全 target セルに掛けるので**薄肉ほど絶対膨張量が大きい**。圧縮比（型開き量比）が設計指標のときに使う。
  - **stroke モード**（`compression_stroke_mm` が None 以外）: `h_eff = h + stroke`。全 target セルに同じ絶対量を加算するので**段差プレートの段差（例: t0.50 − t0.35 = 0.15 mm）が圧縮位相中も保存される**。金型シム量（絶対距離）が設計指標のときに使う。`effective_factor = 1 + stroke · A_cm / V_total`、`A_cm = Geometry.compression_area_mm2()`（圧縮対象セルの**面積** mm²）、`V_total = volume_cm3 × 1000` mm³。
  - どちらのモードでも Film gate / Direct gate のように**プレート本体だけが膨らむ**形状ではランナー・スプルーが膨張に寄与しない分 `effective_factor` が薄まる（実機の挙動と整合）。uniform プレートで `factor = (h + stroke)/h` に揃えると両モードは厳密に等価（`tests/test_compression_stroke.py::test_uniform_plate_stroke_factor_equivalence_for_T_fill` で担保）。
- ウェルドライン: **合流角ベース**（v0.31.0）。各セルで向かい合う隣接ペア（x / y / 対角2本）を見て、両側の流れ方向（`+∇τ` の単位ベクトル、`_flow_direction`）が**そのセルへ向かい**、かつ両側より遅く充填（到着時刻の稜線、落差の偏りが 10:1 以内 = `CREST_BALANCE`、機械精度の同値は 1 セル先で読む `CREST_TIE_RTOL`）なら合流点。2方向のなす角（開き角、180°=正面衝突）を `FlowResult.weld_angle_deg` に持ち、`weld_score_from_angle` が `WELD_MIN_ANGLE_DEG`(0)→`WELD_FULL_ANGLE_DEG`(45) で [0,1] に写す。45° は商用 CAE の「合流角 135°」境界で、以上がウェルド（濃い赤）、未満がメルド（薄い赤、alpha 床 0.35）。壁際セル（8近傍に壁）とゲート周囲2セルは除外。**旧来の「8近傍中6個以上が早い」は局所最大検出で、合流後も流れ続ける線（穴の後ろ）を構造的に見落としていた**（下流3セルが必ず遅い）。描画側 `render_weldlines(weld_min_angle_deg=...)` は角度場を再しきい値化するので、UI スライダー（出力 → メルド表示の下限角、`key="weld_min_angle"`）は解き直し不要 — 解析後の rerun では `_refresh_weld_assets` がキャッシュ結果から weld.png を描き直し、`mfs_settings` と `mfs_zip_bytes` の該当エントリを差し替える（描画は実行ボタンの中にしか無いので、これが無いとスライダーは次回実行まで無反応）。穴の後ろの線は根元数 mm だけウェルドで残りはメルド — 薄板が厚いランプから線状に養われノッチが横流れで埋まる模型の挙動で、等値線の中央の舌と対応する。
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
- `s` は τ に依存し、τ は `S(h_core)` に依存するので **fixed-point 反復**で釣り合わせる：1) baseline (`h_core=h`) で τ_baseline を解く → 2) 体積CDF写像の `t_arrival` から `s_new`、`h_core_new` を計算 → 3) 新しい `S` で τ を再解 → 4) `‖Δτ‖` が `skin_convergence_tol` を下回るか `skin_max_iterations` に達するまで反復。
- **絶対時間スケーリング**: **体積重み付き平均 τ** の比（反復後 / baseline、**同一の still-flowing 集合上**で両方評価）で `T_fill` を比例倍する（圧力一定近似 = 流量が抵抗増分だけ細る）。v0.24 までは frozen 除外の max を全セル max で割っており、分母が病的1セルに支配されていた（Issue #52）— セルが凍ると分子からだけ抜けて比が雪崩れる構造。現行は凍ったセルが両側から同時に抜ける。実測に使った代表値は metadata の `tau_rep_flow` / `tau_rep_baseline`。既定 `skin_max_iterations=20`（旧5。定圧近似の正帰還は設計上雪崩れるもので、雪崩の途中で上限に当たると half-frozen のもっともらしい中間状態を返す）。
- **ショートショット**: 反復後に `h - 2·s ≤ h_min` となったセルを `short_shot_mask` に記録。本来流路が遮断されるが、数値安定性のため `h_core` には `min_core_thickness_mm` 以上のフロアが残る。可視化で赤マーク。
- 出力: `FlowResult.skin_thickness_mm`, `core_thickness_mm`, `short_shot_mask`、metadata に `skin_iterations / skin_converged / T_fill_inflation / tau_rep_flow / tau_rep_baseline / short_shot_cells / short_shot_fraction`。
- 可視化: `render_skin_layer_map(result, path)` でスキン厚マップ、`render_core_layer_map(result, path)` でコア層 + ショートショット。

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
  t_arr ← 体積CDF写像(τ, cell_volume, T_fill)   ← HeleShawSolver と同一の写像 (Issue #52)
  T_k(x,y) ← neumann_layer_temperatures(ζ_centers, t_arr, h_total, T_melt, T_mold, α)  (max(_, T_mold) で clamp)
  γ̇_k(x,y) ← poiseuille_shear_rates(ζ_centers, V, h_total, floor=0.01)
  η_k(x,y) ← cross_wlf_viscosity(material, T_k, γ̇_k, 0)
  S_total ← Σ_k h_total³ m_k / (12 η_k)
  τ_new ← _solve_tau_field(S_total, dirichlet)
  T_fill_new ← T_fill_baseline · (mean_V(τ_new) / mean_V(τ_baseline))   ← 圧力一定近似（体積重み付き平均比、Issue #52）
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
- `metadata` に `solver_kind="multilayer"` / `num_layers` / `layer_distribution` / `layer_zeta` / `layer_moments` / `multilayer_iterations` / `multilayer_converged` / `T_fill_inflation` / `tau_rep_flow` / `tau_rep_baseline` / `damping_factor` / `damping_events` / `T_solid_K` / `short_shot_cells` / `short_shot_fraction` / `shear_heating_enabled` / `shear_heating_max_K` / `shear_heating_mean_K` / `brinkman_number_max` / `brinkman_number_mean` / `specific_heat_J_kgK` / `thermal_conductivity_W_mK`。

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

`tests/` 配下に 26 ファイル、合計 **511 テスト** (510 pass + 1 skip — skip はショートショット 高 threshold ケース)：

- `test_smoke.py` — 4件: import / MaterialDB / build_demo_geometry / Cross-WLF 単調性
- `test_solver_1d.py` — 12件: 1Dストリップの解析解 `τ(x) = x(2L−x)/(2S)` との比較。max誤差 <2%、メッシュ細分化で誤差減少を保証 / **行列の構造** — 非対称エントリがゲート行に限られること、固定点を消去した内部ブロックが対称正定値であること（`docs/流動解析の仕組み_想定問答_技術編.md` が顧客に約束している性質なので、散文だけに置かない）、ゲートに繋がらない成分があるとその保証が崩れること（数学の性質であって、修正後も不変）、**`solve()` がゲート未到達成分を入口で拒否すること（Issue #58 修正。旧実装は特異な純 Neumann ブロックを `spsolve` に渡してもっともらしいゴミを返していた。strict xfail が XPASS に転じたのを機にマーカーを外して現在形の assert に書き換えた）**、全ゲートが mask 外に落ちたときも拒否すること、**対角接触だけの「橋」を連結と数えないこと（到達性チェックは5点ステンシルと同じ4近傍。8近傍でラベリングすると特異ブロックを見逃す）**、ゲート列の消去が解を動かさないこと（消去は厳密）。**mask に穴を開けた形状で組む**のが要点で、mask が全 True だと圧縮行列インデックスと生グリッドインデックスが一致してしまい、両者を取り違えたテストも通る（`Geometry.gates` はグリッド座標、`A` はそうでない）
- `test_geometry_film_gate.py` — 43件: シルエット / 厚み / ゲート土手 / 体積スケール / バリデーション / バランサー（1段スカラー形 + N段ネスト） / プレート分割（ゲート側/反ゲート側2層） / solver 統合 / **compression_mask（プレート本体のみ膨張、ランナー・ゲートは不変）**
- `test_geometry_direct_gate.py` — 26件: シルエット（プレート単体・ランナー無し） / ゲート位置（左右中央＋ゲート側辺から `g_off` mm 内側） / ゲート径 / 体積 / 圧縮マスク（プレート全体） / バリデーション（ゲート円の突き抜けチェック含む） / solver 統合 / 圧縮成形による T_fill 短縮 / **プレート分割（ゲート側／反ゲート側2層、resolved_plate_zones、None フォールバック、バリデーション）**
- `test_geometry_film_gate2.py` — 33件: 直角台形シルエット / 厚み場のプロファイル（連続性・段差の有無・x非依存性）/ ゲート位置可変 / 2段テーパ / 深ランナー / `resolved_*` フォールバック / バリデーション / solver 統合
- `test_geometry_profile_gate.py` — 58件: JSON I/O（round-trip、パス付きエラー、未知キー拒否、非オブジェクトのセクション拒否）/ バリデーション / シルエット（ランド・ランプ式・cap 底打ち・アイランド・外壁・対称性 flip・非対称）/ 井戸（フロア深さ・max 合成・体積差分を放射積分と照合）/ **閉形式の体積検算**（直壁最小スペック ±3%）/ グリッドはみ出し拒否 / compression_mask / プレート2層 / solver 統合 / **溶接ダム（`island.weld`）**（帯内は一定深さ・帯外と島外は不変・体積減を求積と照合・未指定時は旧挙動と bit-identical・round-trip・バリデーション）/ **非有限値の拒否**（`validate` の dataclass 走査が数値リーフ28個に届くこと・JSON 経由 × NaN/±Inf・パーサを通さない直接構築・NumPy スカラが走査から落ちないこと）/ **配列フィールドの構造検証**（5フィールド × 壊れた値8種、2文字文字列のアンパック化けを含む） / **溶接ダム深さ 0 = PL 接触**（帯が mask から抜け、井戸が貫くセルは残り、ソルバが完走）/ **井戸深さが壁の到達範囲 `half_width·tan(wall_angle)` を超えたら拒否**（境界値は通す、壁角 90° は無制限）/ **ゲート土手 vs メッシュ**（`gate_exit_width` がメッシュ間隔を切って最下行が全閉するとき組立段階で拒否・健常幅では連結成分1個で組めること — Issue #58 の実測経路）
- `test_spec_source.py` — 44件: **gitignore ルールそのもの**（リンク本体は実リポに問い、配下は `.gitignore` を写した使い捨てリポで判定 — 実リポに問うと「シンボリックリンクの先」で 128 が返り、**機能を実際に使っている環境でだけ skip する**。ルールが無いと落ちることも別途 assert）/ ルールが広がりすぎてデモスペックを飲まないこと（**使い捨てリポの未追跡コピーに問う** — `git check-ignore` は追跡中のファイルにはルールに関わらず「無視しない」を返すので、実リポの実ファイルに問うと `*.json` でも通る空のガードになる）/ デモスペックが実際に追跡されていること / `spec_root`（不在・実ディレクトリ・外を指す symlink・**完全 resolve**・リンク切れ・ファイル・**symlink ループ**）/ `spec_link_exists`（不在と「あるが壊れている」の区別）/ `list_spec_files`（`*.json` のみ・ディレクトリ除外・**mtime 順でなく名前順**・読めないディレクトリで例外）/ `choose_spec_origin` 10 通り + IO を踏まないこと / **配線を AppTest で実 `app.py` に対して**（リンク無しで一覧が出ない・既定が未選択で何も読まれない・選択で読める・**ドロップが一覧に勝ちその旨が出て一覧が disabled になる**・解除で一覧に戻る・リンク切れの告知・読めないフォルダ/ファイルの告知に**パスが含まれない**・記録される名前がファイル名のみ）
- `test_settings_record.py` — 11件: 形状 config の全フィールドが記録されること（dataclass 自身と突き合わせ）/ `None` が null として残ること / tuple が list になり JSON round-trip すること / **アップロードしたスペックの中身がシリアライズ結果に現れないこと**（キーの有無でなく実際の数値で検証）/ フィンガープリントの同一性・差分検出 / bytes と str の等価 / config dataclass を持たない入力（`cfg=None`）/ `spec` 配下のキーが `name`/`sha256`/`bytes` に限られること（スペック自身の `name` フィールドは中身なので載せない）/ 型・空文字の拒否
- `test_version.py` — 5件: `pyproject.toml` と `__version__` の一致 / CHANGELOG に現行版の項目がある / CHANGELOG の版が降順 / `build_label()` が版で始まる / git メタデータ（SHA・日付・dirty フラグ）の反映
- `test_compression_stroke.py` — 9件: stroke モード後方互換（`compression_stroke_mm=None` で factor モードと完全一致）/ 段差プレートで段差保存 / 全 target セル等量加算 / `stroke=0` で圧縮 OFF 一致 / uniform プレートで factor モードと stroke モードが等価 / metadata の `compression_mode` / `compression_stroke_mm` 露出 / `Geometry.compression_area_mm2()` ヘルパー
- `test_short_shot_timeline.py` — 37件: 凍結セルが T_fill を決めないこと / 未充填セルの `fill_time_s` と `pressure_norm` が NaN / 時間軸が `tau_max_flow` 由来であること（`tau_max` との比が 10 倍超）/ **凍結セルに封じ込められた領域も未充填になること**（連結性）/ 凍結が無ければ挙動不変 / スキン OFF で `unfillable_mask` は None / ゲート凍結で全キャビティ未充填 / `_tau_reference` のフォールバック / metadata の内訳 / **色の場が死んだセルを隣の生きたセルから取ること**（合成データで検証。実部品では隣が最遅セルになりがちで、実物ベースだと差が出ない）/ 死んだセルが `filled` に関係なく覆われ続けること / タイトルのショートショット表記 / **未充填セルにウェルド・エアトラップを立てないこと**（生 τ なら立つことを前提 assert してから検証）/ 有限な充填時間が1種類でもウェルド図が描けること / 圧力マップが未充填セルを専用色で塗ること（アルファ固定後の画素で検証）/ **ゲート以外が全部封じ込められたとき T_fill がベースラインに留まること**（全体最大へのフォールバックは死セルの τ を呼び戻す）/ 描画の時間軸とヘッドラインが一致すること / **色スケール・タイトル・フレーム時刻の3者が同じ軸を読むこと**（欠陥は重複そのものなので3つまとめて assert）/ 反復途中で全セルが凍っても落ちないこと（反復上限3/5/8）/ **生きた領域を切り離して解き直していること**（独立に組んだ live-only 解と 25% 以内、未再解なら 3.3 倍ずれる）/ ベースライン時間が充填する体積だけを数えること / inflation が同じ領域の2状態の比であること / **射出率未指定の既定経路でも体積スケールが効くこと**（他のテストは全部明示的に率を渡していて、この経路を通らなかった） / **live 領域の結果が背後の死体積の量に依存しないこと**（同じ live 形状で死領域を20セル/100セルに振り、`h_core` と `fill_time` がビット一致。スキンの不動点を全体キャビティで1回だけ回すと 3.9 倍ずれる）/ **報告した live キャビティを新規に解き直しても動かないこと**（自己整合の検算）/ 制限後のソルバが射出率を引き継ぐこと / **体積CDF写像**（均一ストリップで到達が体積線形＝セル k が (k+1)/n·T になること — 旧線形写像は中央を 0.75T と返す / 外れ値1セルを足しても他セルの絶対時刻が一切動かないこと（CDF の厳密な不変性）/ `_arrival_time_field` の単体契約: 同値 τ がグループ末尾時刻を共有・除外セルは NaN・最大 τ がちょうど T_fill・空選択で全 NaN / `_tau_volume_mean` の単体契約: 体積重み・空/ゼロ体積/ゼロ τ で None）/ **ゲート脇チョークは到着時計ではほぼスキンを持たないこと**（到着スナップショット意味論のピン留め。ゲート近傍凍結を露光時計 s(T_fill−t_arr) でモデル化する日が来たら、これが変更を記録するテスト） / **ドメインパス安全弁**（`MAX_DOMAIN_PASSES` を 0 に monkeypatch して弁の経路を直接踏み、残った凍結セルが live に残らず unfillable に落ちること・`domain_converged=False` が立つこと。正常経路では True）
- `test_weld_detection.py` — 9件: 一様流で 0 / 対向 2 ゲートで中央行だけ、正面衝突は 1.0 / **穴の後ろに出て、手前（分流）と側面には出ない** / ゲート周囲と壁の外は 0 / 合成 V 字の τ で開き角ランプの値（全開 1.0・平行 0・中点 0.5） / 引数検証 / **角度場の再しきい値化**（緩めると増え、締めても新規セルは出ず、正面セルは両方で 1.0、描画が通る） / 角度場なしの結果は `weld_score` にフォールバック / **偶数幅グリッドで中央2列が機械精度で同値でも両列に線が立つ**（稜線ゲートは tie 許容ぶん負を許す）
- `test_skin_layer.py` — 6件: skin OFF/ON、`c_skin=0` で baseline 復元、極薄肉でのショートショット検出、metadata の整合性
- `test_multilayer_solver.py` — 46件: 層分布プリミティブ (uniform / wall_refined / 端点・対称性・壁細密性・plan 例一致・Σm=1/6) / コンダクタンス helper (N=1 で h³/12η、cavity 外ゼロ、(N,) と (N,ny,nx) η 形状) / N=1 で既存 `HeleShawSolver` と一致 (anchor) / Σh_k=h_total / 後方互換 / wall_refined ソルバー受理 / 温度結合 (layer フィールド populated/None、τ_max 変化、収束性、tol 感度、metadata、壁<中央温度) / ショートショット (metadata 存在、warm で 0、極薄+高 threshold で発火、threshold 0 で 0) / damping (metadata、引数検証、ω=1 動作) / **剪断発熱段階1** (既定 OFF で後方互換、Br 数は常に populated、ON で ΔT_max>0 + 層フィールド shape、ON で η が下がる、material 由来 cp/k メタデータ確認) / **ゲート到達性** (層別 solve も未到達成分を入口で拒否 — base の solve() を経由しないので独立に検査が要る) / **体積CDF整合** (N=1 で base と fill_time が1セル量子以内で一致 / 膨張比が tau_rep_flow/tau_rep_baseline と厳密一致し max 比と判別可能な差を持つこと / 層温度が報告 fill_time の Neumann 解と 0.5K 以内 — 旧線形写像なら 14K ずれる)
- `test_multilayer_thermal.py` — 22件: Neumann 1D (t→0 で T_melt、t→∞ で T_mold clamp、対称性、中央 > 壁、t 単調性、入力検証) / Poiseuille (壁で max、中央 floor、shape、floor=0、引数検証) / **剪断発熱段階1** (shape & 非負、γ̇=0 で ΔT=0、γ̇² スケーリング、t≫τ_thermal で頭打ち、極薄 PP の桁感、shape 不整合検出) / **Brinkman 数** (shape & 非負、γ̇=0 でゼロ、極薄高速で Br>1、k と ΔT の非正検出)
- `test_visualizer_3d.py` — 25件: block anatomy（**3トレースとも flat-top Mesh3d**、天面=各 cavity セル2三角形＝面数 2×cavity・`intensitymode="cell"`）、天面が全 cavity セルを覆う（面数=2×cavity＝侵食ゼロ・天面 Z 有限正・床 Z=0）、**flat-top が対角境界を塞ぐ**（対角バンドで面数=2×cavity）、頂点がゲート中心座標（セル端の隅）、境界壁が PL〜天面を覆う、`aspectmode='data'` で等倍、天面=面ごと/側壁=頂点ごと intensity で coloraxis 共有、**厚み段差が縦の段差として描かれる**（段差プレートで天面 z が両厚みを保持＋PL非接触の段差壁が立つ）、**天面ホバーが場の値を露出**（fill/pressure で per-vertex customdata が読める）、**肉厚配色のガード**（2D の `THICKNESS_CMAP` と一致すること／ramp 全域で明→暗に単調であること／薄肉端と厚肉端のコントラスト比が 3:1 以上あること／colorscale の stop が hex と `rgb(...)` のどちらでも読めること）、**基準そのものの判別力**（`Cividis_r` / `Blues` / `Blues_r` / `Jet_r` / `Bluered_r` / `Greys_r` で期待値を固定。定数を差し替える変異注入は名前一致 assert が先に落ちて判定にならないため、基準を直接テスト）、**stop 判定だけでは区間内の逆行を見逃すこと**（緑→マゼンタの**構成した**反例で固定。組込配色を証人にすると、その配色が変わったときに「サンプリングはもう不要」と誤読される）。各ガードを何故そう組んだかは上の `visualizer.py` 項を参照
- `test_fill_render.py` — 24件: 既定配色が turbo であること / 色スケールが 0..T_fill 固定 / 色の層が完全不透明（アルファに mask を持たせない）/ キャビティ外に NaN が残らない / `_nearest_extend` の3挙動 / **オーバーレイが露出するセルが `mask & filled` と完全一致**（侵食も滲みもゼロ）/ 外側とキャビティ内未充填が別の灰色 / 等時線のレベルが T_fill 固定でフレーム間不変 / 等時線 OFF と空フレーム / 全配色でのレンダリング / フレーム PNG が自前のカラーバーを持つこと / **等時線が一度だけ描かれ zorder でオーバーレイの下に入ること**（呼び出し順ではなく zorder で検証。matplotlib は imshow=0 / contour=2 なので、後から呼ぶだけでは隠れない）/ 要求した本数ちょうどが描かれること / 1セル幅キャビティで等時線を諦めること / ゲートマーカーがオーバーレイより上に来ること / **60枚出力しても contour が1回だけであること**（figure を毎フレーム作り直すと出力は同一なので、contour 呼び出し回数を数える以外に検出手段がない）
- `test_fill_player.py` — 18件: フレーム時刻の単一ソース契約（GIF・PNG 連番・プレイヤーの三者一致）/ 充填率の単調性と末尾 1.0 / payload の埋め込みとデコード / オフライン自己完結（`http(s)://` を含まない）/ 入力検証 / ネイティブ幅キャップと component 高さの導出 / **単体 HTML 化**（`<meta charset>` が最初の非 ASCII バイトより前、文書完全性、フラグメント同一性、title/note のエスケープ）
- `test_visualizer_layer.py` — 15件 (1 skip): `render_layer_map` 4 field smoke / 不正 field / 範囲外 layer_idx / thermal_off で field 別動作 / `render_layer_grid` / `render_short_shot_map` (flagged あり/なし、後者は skip 想定可) / `_scalar_layer_field` helper / ζ レンジが metadata に乗ること / **`THICKNESS_CMAP` が明→暗で走ること**（薄肉が明るい側）と**薄肉端が彩度を保つこと**（HSV 彩度 > 0.5。白に寄る単色マップを弾く）/ 層別 thickness パネルが同じ ramp を使うこと
- `test_two_phase.py` — 32件: 不変条件（Ω₁ ⊆ Ω₂ / **最終形状が計量体積をタイ群粒度で保持**（超過ゼロ）/ **ネストは Ω₁ のみ厳密**（Ω₂ は各計量が自分のプール境界で τ₂ を解くため分岐形状で保証外 — この形状での成立は回帰ピンとして残す）/ フルショットで完全充填 / **V_fin ≤ V_shot < V_open（ICM 通常運転）でも圧縮順序を報告**（final_complete でも τ₂ を解いて progress を全前進セルに載せる）/ ゲートは常に Ω₁ / プール外の到着時刻 NaN）/ **ストリップ解析解**（前線位置 n1 = V/(dx²·h_open)、圧縮後 n2 = n1·h_open/h_fin の純算術 / 到着時計の体積線形性 / 前進の連続性 / progress の単調性と終端1）/ 退化と拒否（ICM OFF・stroke 0 で前進なし / **ギャップ不変なら残余 budget があっても τ₂ を解かない**（非一様ストリップ、tau2 is None で判定）/ factor モード受理 / スキン層拒否 / 非正体積拒否 / **ゲート群を覆えない計量の拒否** / **開ギャップを縮める圧縮設定の拒否**（負 stroke・factor<1 は h_open < h_fin で達成体積が計量超過に化ける） / ゲート無し拒否）/ metadata 契約 / **タイ原子性**（2×n 双子ストリップ: 5.5列分の計量が11半セルでなく5全列で止まる）/ **等圧ソース契約**（tau2 が Ω₁ 全セルで 0、外で正）/ 描画（部分/完全ショット両方 / **色のピクセル数比**が領域セル数比に整合 — 凡例パッチが色の存在を常に供給するので「色がある」だけの検証は空 / **凡例が axes の外**（figure legend の bbox と axes bbox の非重複を draw 後に検証）/ **frame_states の契約**（単調成長・最終フレーム = Ω₂・相の分割と時計・前進なしなら射出のみ・num_frames 検証と小予算耐性・**1枚だけの射出相はプール完成時点を見せる** — `linspace(0,T,1)==[0.0]` で空キャビティ→最終形状に飛ぶ穴）/ アニメ GIF のフレーム数）
- `test_film_gate1_ui.py` — 8件 (AppTest): 既定スライダーが組む spec が hamoko spec と一致（導出量の床範囲は `深さ/tan60°` で検算）/ 既定形状が spec を直接ビルドした `Geometry` と mask・肉厚 bit 一致 / アイランド・井戸 OFF で spec の該当節が `None` / **井戸深さの上限が `半幅·tan60°` に追従**（超えると spec は受理・記録されるのに浅く描かれる — Codex P1）/ **バルブ位置の上限がポケット終端 − 半径**（外れるとビルダーが最寄りセルへ黙ってスナップし、記録と注入位置がずれる — Codex P1。井戸 OFF で上限が外壁終端へ戻ること、ゲートセルが記録位置にあること）/ 均一肉厚分岐（split=0・lower/upper が None・プレート肉厚が1種）/ 床が潰れる短い井戸で `floor_t_range=None` / **幅方向ガードの発火**（スライダーでは到達不能なので `core.build_profile_gate_geometry` を monkeypatch してオリフィス下の全セルを mask から抜き、`st.error` に化けて `mfs_geom` が残らないことを固定）
- `test_film_gate2_ui.py` — 7件 (AppTest): 既定スライダーが組む spec が weld spec と一致（肉盗み境界は t=ランド長で同じ直線上、井戸壁角 71.6°、床範囲 1.5 内側）/ 既定形状が spec 直読みと mask・肉厚・ゲート bit 一致 / 水平帯が井戸の外で一様 0.1 / **PL からの距離 0 で帯が cavity から抜け、残りは不変、解析は完走**（穴の周りを回る）/ 水平部 OFF で `weld` が null / 開始 t の範囲がランド長〜肉盗み終端 − 0.5 に追従し終端が肉盗み終端に固定 / 井戸深さ上限がこの図面の壁角（半幅 1.0 → 3.0）
- `test_film_gate3_ui.py` — 7件 (AppTest): 既定スライダーが組む spec が 2bai spec と一致（`symmetric=False`、バルブ t=20.0 は井戸中央でなく図面値）/ 既定形状が spec を直接ビルドした `Geometry` と mask・肉厚・ゲートセル一致 / ゲートセルが w=0 端（出口左端）に集まること / **スライダー値が FG1/FG2 間で漏れないこと**（全 widget に `tag_` 付き key — key 無しだとラベル＋パラメータ同一の widget が値を引き継ぐ、Codex P2）/ **片側・井戸 OFF・0.4mm メッシュでバルブが拒否されないこと**（中心セル判定だと w=0 境界で floor が外に落ちる、Codex P2）/ **片側の井戸半幅の上限が端の余白 `pad + (Wp − 出口幅)/2` に追従**（超えるとビルダーの grid overhang 拒否）/ 片側版のガード発火（monkeypatch、`symmetric=False` 分岐の x_valve）
- `test_weld_ui.py` — 1件 (AppTest): **解析後にメルド下限角スライダーを動かすと、解き直さずに weld.png が再描画され、settings と ZIP 内の weld.png / settings.json が差し替わること**（`mfs_result` が同一オブジェクト、ZIP に重複エントリなし、他の同梱物は残る — Codex P2）
- `test_two_phase_ui.py` — 5件 (AppTest): **UI 既定が二相 ON・ICM ON 0.50・壁面冷却なし**（v0.29.0）/ OFF に戻せば何も走らない / ON + ICM で map 生成・settings.json 記録・ZIP に `two_phase_short_shot.png` と `two_phase_metadata.json` 同梱 / 壁面冷却モデル選択時は警告してスキップ / **solver の ValueError が警告+スキップに化けること**（monkeypatch 注入 — app.py は毎 run `from core.two_phase import` を再実行するのでモジュール属性の差し替えが効く。実ジオメトリで拒否が出るかはゲート群体積依存なので配線だけを固定）。**AppTest の number_input は min_value 未満の set_value を黙って無視する**（clamp でも例外でもなく既定値のまま）。**expander 内の caption は AppTest の平面リストに出ない**ので、描画は session_state のパスとファイル実体で検証する

新機能を足したら**該当する系統のテストファイルにテストを追加**するのが慣例。形状なら `test_geometry_*.py`、solver の挙動なら `test_solver_*.py` か `test_skin_layer.py` か `test_multilayer_solver.py`、純関数の helper なら `test_multilayer_thermal.py`、3D 系なら `test_visualizer_3d.py`、層別可視化なら `test_visualizer_layer.py`、二相ショートショットなら `test_two_phase.py`（UI 配線は `test_two_phase_ui.py`）。

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

`app.py` のサイドバー入力は5系統（`Film gate 1` / `Film gate 2` / `Film gate 3` / `Direct gate` / `Profile gate`）。**Film gate 1 は v0.28.0、片側版（現 Film gate 3）は v0.29.0、Film gate 2 は v0.30.0 で差し替えた** — 旧「台形＋半円ランナー＋▽肉盗み」（`FilmGateConfig` / `build_film_gate_geometry`）と旧「ゲート位置可変」（`FilmGate2Config` / `build_film_gate2_geometry`）は **CLI / ライブラリ専用**に降格し、UI の Film gate 1 / 2 / 3 はスライダーで `GateProfileSpec` を組み立てて Profile gate と同じ `build_profile_gate_geometry` に流す。3 つは共通ヘルパー（`_profile_gate_sidebar(tag, symmetric, defaults)` → `_profile_gate_from_inputs` → `_build_film_gate`）で、違いは `symmetric` と `_ProfileGateDefaults` だけ。`_FILM_GATES` dict（ラベル → `(tag, symmetric, defaults, 記録名)`）がラジオの選択肢・サイドバー分岐・`build_geometry` の分岐の単一ソース。

| ラベル | tag | symmetric | 既定の図面 | 特記 |
|---|---|---|---|---|
| `Film gate 1 (扇状/肉盗み1)` | `f1` | True | `hamoko_gate_furiwake_20260703` | 井戸壁角 60°、バルブ = 井戸中央 |
| `Film gate 2 (扇状/肉盗み2)` | `f2` | True | `hamoko_gate_furiwake_weld_20260818` | 肉盗み下流 t=7〜17 が溶接で水平（`island.weld` 残り厚 0.1）、外壁開始 t=5、井戸壁角 71.6°。図面の肉盗み境界線は t=0 始まりだが UI は t=ランド長で固定するので、同じ直線を t=1 で読んだ 47.64 を既定に置く（1.0mm 格子は spec 直読みと bit 一致） |
| `Film gate 3 (片側/二倍流動長)` | `f3` | False | `hamoko_gate_2bai_20260703` | バルブは w=0 端、幅はバルブ側端から測る、バルブ t = 図面値 20.0 |

既定値の実寸を公開リポに載せることはユーザー承知の上。UI 既定入力は Film gate 3（`index=2`）。スライダーは主要寸法だけで、導出量は図面の結び付きで固定する：肉盗み境界線の t 端点 = `(ランド長, 肉盗み終端)`、外壁開始の幅 = `出口幅/2`（対称）／ `出口幅`（片側）、井戸の床範囲 = `t_range` を `深さ/tan(壁角)` ずつ内側に寄せる（壁角は図面ごとの定数 `_ProfileGateDefaults.well_wall_angle_deg`、床が潰れるなら `None`）、水平部の終端 = 肉盗み終端、バルブ位置の既定 = 井戸中央（`valve_t=None`）／ 図面の値。**用語**: UI では `island` を「肉盗み」、`orifice_diameter` を「バルブゲート径」と呼ぶ。JSON キーとコード識別子は英語のまま（既存スペックを壊さない）。**水平部（溶接ダム）**は肉盗み ON のとき全 Film gate に出るチェックボックス（既定 ON は 2 だけ）で、「PL からの距離（残り流路厚）」は 0 まで下げられる — 0 は鋼材が PL に接した状態で、`validate()` は受理し、ビルダーはその帯を mask から外す（厚み 0 のセルを cavity に残すと `S=0` で特異系。井戸が貫くセルは `in_well` 経由で井戸の深さのまま cavity に残る）。**UI 側で先回りする制約が2つ**: 井戸深さの上限 = `半幅·tan(壁角)`（`validate()` にも同じ制約がある）、バルブ位置の範囲 = `[半径, ポケット終端 − 半径]`（ポケット終端 = 外壁終端と井戸終端の遠い方。外れるとビルダーが最寄りマスクセルへ黙ってスナップする）。幅方向の外れ（ゲート円がポケット先端より広い）はスライダーで表せないので、ビルド後に**ゲート円と mask の交差**（`_valve_orifice_hits_pocket`、ビルダーがスナップ前に見るのと同じ判定）を検査して空なら `ValueError` にする。中心セル判定は片側で w=0 境界上の中心が floor で外に落ちて誤拒否する。settings.json には組んだ spec を `gate_profile` キーで全量記録する（スライダー値であって図面ではないので、Profile gate の fingerprint 方針の対象外）。既定のまま実行した形状が各 spec を Profile gate で読んだ結果と bit 一致することは `tests/test_film_gate{1,2,3}_ui.py` が守る。widget key は `f1_` / `f2_` / `f3_` で分離。**画像入力（PNG/JPG 二値化）は v0.23.0 で撤去した** — 一度も使われず、PDF から JSON スペックを起こす経路が確立して役目が無くなったため。`core.geometry_from_image` ごと消したので、戻すなら git 履歴から拾う。`run_demo.py` の `DEMO_CASES` / `FILM_GATE_CASES` / `DIRECT_GATE_CASES` も同じパラメータ群を扱う（`DEMO_CASES` は CLI 専用、UI からは消した）。**新パラメータを `HeleShawSolver` / `FilmGateConfig` / `DirectGateConfig` に足すなら、UI と CLI の両方に反映する必要がある**。

特に `FilmGateConfig` の `D_flat + D_slope = D` 制約は、UI 側では「`D_flat / D` の比率スライダー」で表現してこの制約を自動満足させている（`app.py` の `flat_ratio` 変数）。CLI 側は直接 `D_flat` / `D_slope` を渡すので、case 定義時に和が `D` になることを手動で保証する必要がある。

その他の UI ↔ CLI ブリッジ方針：

- **バランサー**: UI は段数 N をスライダーで選び、`balancer_base_widths_mm` / `balancer_thicknesses_mm` のタプルを構築して渡す。スカラー形（1段固定）は CLI の旧ケース互換のため温存。CLI で N段にするなら直接タプルを書く。
- **プレート分割**: UI は「段差位置 [mm]」スライダーで `plate_split_height_mm` を出し、値が `0` のときはゲート側／反ゲート側の肉厚スライダーを隠して `plate_thk` 1本に統合、cfg には `plate_lower_thk_mm = plate_upper_thk_mm = None` を渡して uniform モードに落とす。CLI で uniform にしたいときも同じく `plate_split_height_mm=0` ＋ `plate_lower_thk_mm = plate_upper_thk_mm = None` で足りる。
- **スキン層**: UI のトグルが `skin_layer_enabled`、スライダーが `skin_growth_constant`（`c_skin`）。CLI 側はソルバー kwargs に直接渡す（`run_demo.py` の `PP_skin_layer` 参照）。
- **射出条件の単位系**: `injection_velocity_mms` / `injection_volume_flow_cm3s` の既定値は **実機ユニット域**（`eae5394` で再スケール済）。新ケースを足すときも実機相当の値を入れる前提で考える。CLI 既定値（`run_demo.py`）と UI 既定値（`app.py`）はこの方針で揃えてある。
- **Direct gate**: UI では「製品幅 / 製品高 / 段差位置 / ゲート側肉厚 / 反ゲート側肉厚 / ゲート径 / ゲート位置」のスライダー群で `DirectGateConfig` を組み立てる。プレート分割の挙動は Film gate と同じ（段差位置 0 で uniform、`> 0` で 2 層化、cfg には `plate_lower_thk_mm` / `plate_upper_thk_mm` を渡す）。デフォルト値は Film gate と揃えてある（Wp=300 / Hp=50 / 段差=20 / lower=0.35 / upper=0.50 / Φ=3 / g_off=20）。ゲート位置スライダーの上下限はゲート径とプレート高さに連動して動的に計算（突き抜けバリデーションを UI 側でも防御）。CLI でも `DirectGateConfig` の引数に同じ制約がある。
- **圧縮成形のスコープ**: 圧縮で膨らむのは `Geometry.compression_mask` が True のセルだけ。Film gate のビルダーは「プレート本体だけ True」（ランナー・スプルー・ゲートは膨張しない）、Direct gate のビルダーは「プレート全体 True」（cavity = プレート単体なので全部膨張）。`build_demo_geometry` は `compression_mask=None`（旧挙動＝全セル膨張）。新しい形状ビルダーを足すときは「製品本体だけ True」の compression_mask をセットすること。
- **圧縮量の指定方式**: UI は ICM ON 時にラジオで `factor` / `stroke` を選ぶ。factor 選択時は「初期隙間倍率 h_init/h_final」スライダー（`compression_stroke_mm=None` で solver に渡る）、stroke 選択時は「圧縮ストローク [mm]」スライダー（`compression_factor=1.0` ＋ `compression_stroke_mm=<値>` で渡る）。CLI 側は両方を `make_solver()` の kwargs に直接渡し、ケース定義で片方だけ指定する（factor モードならデフォルト、stroke モードなら `compression_stroke_mm=0.70` 等を明示）。段差プレート（plate_lower_thk ≠ plate_upper_thk）の圧縮シミュレーションでは **stroke モード一択**（factor モードだと段差が崩れる）。`FilmGate_PP_stepped_stroke` がこの想定の CLI 参照ケース。
- **壁面冷却モデル**: UI ヘッダー「壁面冷却モデル」のラジオで `なし` / `スキン層` / `層別` の 3 択。**排他選択**により skin と multilayer の同時 ON は構造的に不可能。**UI 既定は『なし』**（v0.29.0。二相ショートショットを既定 ON にしたため — 二相は『なし』専用）。層別を選んだときの既定は **N=7 / wall_refined / max_iter=12**（極薄 t<0.5mm 向け）。
  - **なし**: 既存 `HeleShawSolver`、温度結合なし。
  - **スキン層**: `HeleShawSolver(skin_layer_enabled=True, skin_growth_constant=c_skin, ...)`。Stefan/Neumann 1 層モデル。
  - **層別**: `MultilayerHeleShawSolver(num_layers=N, layer_distribution="wall_refined", thermal_coupling=True, ...)`。Cross-WLF 結合 N 層モデル。スライダーで `num_layers` (3..9、既定 7) / `layer_distribution` (`wall_refined` 既定 / `uniform`) / `max_iterations` (1..20、既定 12) / `convergence_tol` / `solidification_temperature_fraction` を出す。**剪断発熱補正 (段階1)** はチェックボックス `shear_heating_enabled` (極薄向け既定 ON)。
  - CLI 側は `_solve_and_export(multilayer=True, num_layers=..., layer_distribution=..., shear_heating_enabled=..., ...)` で明示。`skin_layer=True` と `multilayer=True` の同時指定は `ValueError`。`FilmGate_PP_multilayer_5L` が層別の参照ケース、`FilmGate_PP_multilayer_5L_shear` が剪断発熱 ON の比較ケース (高 V、N=7、極薄)。
  - 結果ペインに「層別プロファイル (Multi-layer N=...)」expander が現れ、温度グリッド / 粘度グリッド / ショートショットマップを表示、各 PNG ダウンロード + ZIP exports に同梱。**剪断発熱メタデータ** (ΔT_max / ΔT_mean / Brinkman 数 max & mean、信号灯 🟢/🟡/🔴) は expander 直下のキャプションに出る。
- **二相ショートショット**: UI はサイドバー「ショートショット（計量制限）」expander の checkbox（`key="two_phase_on"`、**v0.29.0 から既定 ON**。ICM も既定 ON・ストローク 0.50 mm）＋計量体積 number_input（`key="two_phase_shot_volume"`、cm³）。壁面冷却モデルが『なし』以外だと警告してスキップ（モデルの用途定義: 計量律速は凍結を含まない）。結果ペインに専用 expander（マップ + 二相アニメ GIF + 計量体積 / 射出終了時充填率 / 圧縮後充填率）、ZIP に `two_phase_short_shot.png` / `two_phase.gif` / `two_phase_metadata.json`、settings.json に `two_phase_short_shot` セクション。**壁面冷却が『なし』以外なら checkbox 直下に常時警告**（実行時の一過性警告だけだと rerun で消え「ON にしても何も起きない」に見える — UI 既定の壁面冷却は層別なので、これが無いと既定状態で機能が死んで見える）。スキップ理由は `mfs_two_phase_skip` に永続化して結果ペインに st.info で残す。計量体積入力の下に前回実行形状の体積目安（最終 / 開きギャップ）。CLI は `_solve_and_export(two_phase_shot_volume_cm3=...)`（skin / multilayer と排他、`FilmGate_PP_two_phase_short` が参照ケース）。
- **ゲートスペックの読込 (`core/spec_source.py`)**: 「スペック入力」ラジオは `デモプリセット / ローカルから読込 / JSON貼り付け`。分岐は `SpecMode` enum で行う（旧実装はラベルの `startswith` を見ていたので、文言を直すと挙動が変わった）。**ローカル一覧が出るのは、リポ直下の `local_specs` がディレクトリに解決するときだけ**。これは gitignore 済みの名前に**固定**してあり、設定で変えられない。理由は2つ：(1) 環境変数だと**フェイルオープン**で、Streamlit Cloud は secrets のトップレベル `str`/`int`/`float` を `os.environ` に昇格させるため、ローカルの `secrets.toml` を設定欄に貼るだけで公開インスタンスにファイル読み取り口が生える。gitignore されたパスは checkout + secrets で組まれる Cloud 上に存在させる手段が無い。(2) root を設定可能にすると、リポ内を指して `git add -A` するのが**顧客寸法を public リポに commit する最短経路**になり、env ゲートもパス閉じ込めもこれには効かない。固定すれば構造的に起きない。**テキストのパス欄は無い** — ユーザー入力のパスが存在しないので `..`／`~` 展開／prefix 一致／拡張子縛りの問題が書き忘れではなく発生しない。一覧の既定は `— 未選択 —` センチネル：`build_geometry()` は実行ボタンの後ろではなく**毎 rerun 走る**ので、既定で実ファイルを選ぶとページを開いた瞬間に顧客形状が描画される。ドロップと一覧が同時に生きているときは**ドロップが勝ち**、勝った側をサイドバーに明示して負けた一覧を `disabled` にする（帯はメインカラムでなくサイドバーに出す — 食い違っているウィジェットの隣でなければ届かない）。**例外の文言を画面に出すな**：絶対パスが顧客名と案件名を含み、`client.showErrorDetails` の既定は `full`。`Path.resolve()` は symlink ループで `OSError` でなく `RuntimeError` を投げ、`Path.glob()` は権限エラーを飲み込んで空を返す（このため一覧は `os.scandir`）。CLI 側にこの読込経路は無い（`run_demo.py` はケース定義に直接スペックを書く）。
- **結果 ZIP の中身**: 「⬇ GIF + フレーム画像をダウンロード」は `fill.gif` / 各マップ PNG / `frames/` の連番 PNG / `metadata.json`（解析結果）/ **`settings.json`（入力設定）** に加えて **`player.html`** を含む。`settings.json` は形状 config の全フィールド・材料・射出条件・壁面冷却・圧縮・出力・版を持つ（`core/settings_record.py`）。**アップロードしたスペック JSON は名前と SHA-256 だけを記録し、中身は載せない** — ZIP は人に渡す前提なので、実図面由来の寸法を同梱しない。UI に埋め込んでいるのと同じプレイヤーを `wrap_standalone_html()` で完全な HTML 文書に包んだもので、ダブルクリックすれば追加ソフト無しでコマ送りできる。**フラグメントをそのまま `.html` として出すな** — `<meta charset="utf-8">` が無いと `file://` で開いたブラウザが CP932 等にフォールバックして日本語ラベルが化ける（`tests/test_fill_player.py` が charset の位置を検証している）。同梱で ZIP はおよそ 2 倍になる（base64 は 4/3 に膨らみ、PNG は圧縮済みで deflate が効かない）。
- **剪断発熱補正 (viscous dissipation, 段階1)**: 層別モード専用。`ΔT_shear,k = (η_k·γ̇_k²)·min(t_arr, τ_thermal)/(ρ·cp)`、`τ_thermal = h²/(π²·α)`。fixed-point ループで前イテレーションの `η_k` から ΔT_shear を計算 → Neumann 温度に加算 → Cross-WLF で η 再評価。負のフィードバック (T↑ → η↓ → 発熱↓) なので発散しにくい。**Brinkman 数 `Br = η·γ̇²·h²/(k·ΔT_ref)` は補正 OFF でも常に計算**してメタデータに出すので、必要性を事前判定できる。Br>2 は段階2 (1D FDM 陰解法) が本来必要なシグナル。材料 DB 拡張: `specific_heat_J_kgK` 追加 (8 樹脂)、熱伝導率は `k = α·ρ·cp` で派生。
