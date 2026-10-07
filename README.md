# mold-flow-sim

日本語 | [English](README.en.md)

[![CI](https://github.com/shostako/mold-flow-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/shostako/mold-flow-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

射出成形の流動解析を Hele-Shaw 近似 + Cross-WLF 粘度モデル + Pseudo-Conduction 法で
大きく簡略化した Python シミュレータ。薄肉プレートとそのゲートの初期検討・教育・概念検証用。

> **Moldflow / Moldex3D の代替ではない。** 面内の 3D 流れ、保圧、結晶化、収縮・反りはモデル化していない。
> 厚み方向の温度と固化はスキン層／層別モデルで近似しているだけである。実際の金型設計の判断には商用 CAE を使うこと。

変更履歴: [CHANGELOG.md](CHANGELOG.md)

## デモ

Streamlit Community Cloud の無料枠で動かしている。ブラウザだけで触れる。

<https://mold-flow-sim.streamlit.app>

> 無料枠なので、しばらくアクセスが無いと初回は起動に 30 秒〜1 分かかる。
> 計算は Streamlit Cloud の共有 CPU（1 vCPU / 1 GB RAM）で走るため、細かいメッシュで層別モデルを回すと数十秒かかることがある。

## できること

**解析**

- 2D 構造格子上で、薄肉キャビティの疑似充填時間場 τ を楕円型方程式 −∇·(S∇τ)=1 の一発解で求める
- 壁面冷却モデルを 3 つから選ぶ（画面の既定は層別、N=7）
  - なし: 等温・代表粘度のみ
  - スキン層: Stefan/Neumann の壁凍結フロント `s(t) = c_skin·√(αt)` を取り込み、コア層 `h_core = h - 2s` だけが流れる。先端が通った後も壁は冷え続ける（露光時計）。封止したセルとその先の未充填を出す
  - 層別: 厚み方向を N 層に分け、Neumann 1D の温度分布と層ごとの Cross-WLF 粘度を固定点反復で結合する。中央層の温度でショートショットを判定する。剪断発熱の補正（段階1、閉形式の局所近似）と Brinkman 数の診断つき
  - 層別の充填の解き方は 2 つから選ぶ（v0.62.0）。既定は従来の「一発で解く」（充填の順番を τ の 1 回の計算で決める）。「時間で追う」は射出した量だけ先端を時間で進め、ゲートブロックが埋まりきる前に製品の中央から入り始める前半の動きを出す（計算は 20 倍ほどかかる）
- 射出圧縮成形 (ICM) の等価モデル（圧縮ストロークで流路を広げて充填時間を短縮する）
- 二相ショートショットモデル: 計量を絞ったショートショットを、成形機の条件のまま予測する。射出相（型開きギャップで計量体積まで充填）と圧縮相（型を閉じて溶融プールを前進させる、体積保存）を線形求解 2 回で解く。射出相にはスキン層・層別モデルを乗せられる
- 射出条件は「射出率を直接入力」か「スクリュー径と射出速度から計算」（既定、多段射出と V/P 切替位置にも対応）

**形状**

- Film gate 1〜15: 肉厚調整フィルムゲート（ランド・メインランプ・肉盗み・井戸・外壁線を持つゲートブロック）のパラメトリック入力。実際の図面や改修案から取った 15 通りの既定値を持つ
  - 扇状（両側）: 肉盗み、縁部深彫り、ランプ奥の削り込み、ランド両端の増厚、斜面角度の徐変、ランド長可変（コートハンガー型）
  - 片側（流動長 2 倍）、振り分け（ミニ扇×2、L 字ランナー）、T 字ランナー
  - 画面の表示名に付く「0807」などの数字は、元にした図面の日付
- Direct gate: 単純なプレート＋直接ゲート
- Profile gate: 図面から起こした JSON スペック（複数の扇とランナーにも対応）

**結果**

- 充填アニメーション（コマ送りできるプレーヤー）、等時線
- 相対圧力マップ、ウェルド／メルドライン（2 つの流れが出会う角度で判定）、エアトラップ候補
- スキン層・層別モデルの層マップ、二相ショートショットの履歴アニメーション
- 3D 表示（Plotly）
- 結果一式の ZIP（画像、`metadata.json`、入力条件の `settings.json`、単体で開ける `player.html`）
- ゲートブロックの図面を A3 PDF で自動作図（平面図 1:1、断面 4 本、寸法線と注記つき）。Film gate 1〜15 と Profile gate のみで、Direct gate には出ない
- ゲートブロックの樹脂側を IGES（3D）で書き出す（同じく Film gate 1〜15 と Profile gate のみ）。書き出したファイルを読み戻してソルバーの形と網目単位で照合し、一致したときだけダウンロードに出す。CAD カーネル（OCP）が要るので手元の環境専用で、公開版のデモには無い

材料は PP / PP_T10 / PP_T20 / PP_T30 / ABS / PC / PA66 / PMMA（`data/materials.json` の generic 値）。

## できないこと（既知の制限）

| 項目 | 状態 |
|------|------|
| 面内の 3D 流れ・ジェッティング・コーナー渦 | 未対応（Hele-Shaw 系の根本的な限界。完全 3D の FVM/FEM が要る） |
| 保圧（パッキング段階） | 未対応。充填までしか解かない |
| 結晶化・収縮・反り・残留応力 | 未対応 |
| 剪断発熱 段階2（厚み方向のエネルギー方程式） | 未対応。段階1 は閉形式の局所近似で、Br ≫ 1 の領域でずれる |
| 流動場から剪断速度への帰還 | 未対応。γ̇ は層とセルごとに分布するが、元は射出速度の代表値 1 つ |
| 圧縮相の実時間 | 未対応。圧縮相は前進の順序だけを出す。圧縮中の凍結、射出と圧縮の重なりも扱わない |
| 絶対圧力場・必要型締力 | 未対応。圧力は正規化値（ゲート = 1、先端 = 0）のみ |
| 層内の対流項 | 未対応。1D Neumann は純粋な拡散のみ（極厚 h > 4 mm では崩れる） |
| STL / STEP の直接読込 | 未対応。パラメトリック形状と JSON スペックのみ |
| 中立面メッシュ（非構造格子） | 未対応。構造格子のみ |

## ロードマップ

済み:

1. 基盤整備 — パッケージ化、CI、テスト、1D 解析解との比較検証
2. 形状入力 — Film gate 1〜15、Direct gate、図面由来の JSON スペック（Profile gate）
3. 過渡熱と固化層 — スキン層モデル、層別 N 層モデル、剪断発熱 段階1
4. ICM の二相化（部分）— 二相ショートショットモデル。通常の充填時間・圧力マップは今も等価モデルのまま
5. 成形機の条件からの入力 — スクリュー径・射出速度・多段射出・V/P 切替
6. 出力の実用化 — 結果 ZIP と入力条件の記録、ゲートブロック図面の自動作図、IGES 書き出し

残り:

1. 流動場から速度場への帰還 — 解いた τ から局所流速を起こして γ̇ に戻す
2. 数値の地盤強化 — 反復法（CG / AMG）、メッシュ収束テスト
3. 入出力 — VTK 書き出し、材料 DB の出典明記と拡張
4. ICM の続き — 二相モデルを本線に接続し、圧縮相に型締め速度由来の実時間を与える。圧縮中の凍結はその先
5. 剪断発熱 段階2 — エネルギー方程式の厚み方向 1D FDM 陰解法

## インストール

```bash
git clone https://github.com/shostako/mold-flow-sim.git
cd mold-flow-sim
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # 実行に要るものだけ
pip install -e ".[dev]"              # 開発用（ruff, pytest を含む）
pip install -e ".[cad]"              # IGES 書き出しを使うとき（OCP、約 160 MB）
```

Python 3.11 以上が要る。

> `requirements.txt` / `runtime.txt` / `packages.txt` / `.streamlit/config.toml` は Streamlit Community Cloud 用のデプロイ設定で、
> ローカル開発では使わない（正本は `pyproject.toml`）。

## 実行

### Streamlit UI

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開く。左のサイドバーで形状・材料・射出条件・壁面冷却を設定し、「解析実行」を押す。

### CLI バッチ（パラメータスイープ）

```bash
python run_demo.py
python run_demo.py --cases PP_baseline PP_dual_gate
```

`outputs/<case>/` に GIF・PNG・連番フレームを書き出す。

### テストと lint

```bash
pytest tests/
ruff check . && ruff format --check .
```

## 物理モデル概要

### Hele-Shaw 近似 + Pseudo-Conduction 法

充填過程を時間進行ではなく、楕円型問題の一発解で解く:

```
−∇·(S ∇τ) = 1    (キャビティ内)
τ = 0            (ゲート, Dirichlet)
S∇τ·n = 0        (壁面, Neumann)
S = h³ / (12·η_eff)
```

符号は実装の離散化に合わせてある（対角 `+Σcoeff` / 非対角 `−coeff` / 右辺 `+1`）。
連続形で書けば `∇·(S∇τ) = −1`。

この符号の取り方で、制約を入れる前の作用素は対称かつ半正定値になる（面コンダクタンスを両隣で共有するため）。
ただし組み上がった `A` は対称でも正定値でもない。Dirichlet 条件を行にしか適用しておらず、
ゲート行を単位行に潰す一方で、隣の内部行はゲート列の `−coeff` を残したままだからだ。
ロードマップの CG / AMG 化には、先にゲート列を消去する必要がある。消去自体は近似ではなく厳密
（ゲートで `τ = 0` なので右辺に移る項がゼロ）だが、`spsolve` は対称性を要求しないので今は手を付けていない。

- `τ` は擬似到達時間場（ゲートからの「距離」の単調関数）
- 絶対時間への換算は体積の累積分布で行う: `fill_time(x,y) = T_fill · V(τ' ≤ τ(x,y)) / V_solved`。
  定率射出なら先端は体積に比例して進むので、τ の順に体積を積算した値が到達時刻になる。
  `V_solved` は充填できるセルの体積で、封止で切られた未充填セルは含まない
- `T_fill = V_solved / Q` は定率射出・圧縮なしのとき（壁面冷却なし、またはスキン層の速度制御時計）。
  射出圧縮の等価モデルは充填時間を短縮し、スキン層の圧力一定時計と層別の熱結合は、体積で重み付けした τ の比で再スケールする
- 面コンダクタンスは隣接セルの調和平均

### Cross-WLF 粘度モデル

```
η(γ̇, T, P) = η₀(T,P) / (1 + (η₀ γ̇ / τ*)^(1−n))
η₀(T,P) = D₁ · exp(−A₁ (T−T*) / (Ã₂ + (T−T*)))
T* = D₂ + D₃ P
```

材料パラメータは `data/materials.json`（generic 値）。

### 層別 Hele-Shaw ソルバー

厚み方向を `N` 層に分け、各層に温度・粘度・剪断速度を持たせる。Neumann 1D の重ね合わせで層ごとの温度を求め、
Cross-WLF で層ごとの粘度に変換し、厚み方向に粘度が変わる流れの潤滑の積分でコンダクタンスにまとめる（層の重みは中央面からの距離の 2 乗）:

```
T(z, t) = T_mold + (T_melt - T_mold) · [erf(z/(2√(αt))) + erf((h-z)/(2√(αt))) - 1]
γ̇_k(x,y) = (6V/h) · |2ζ_k - 1|                               # Poiseuille 解析微分
η_k(x,y) = cross_wlf_viscosity(material, T_k, γ̇_k, 0)
S_total(x,y) = ∫(z − h/2)²/η dz = h³ · Σ_k m_k / η_k          # m_k = [(ζ − 1/2)³/3]、Σ m_k = 1/12
```

`τ ↔ T_k ↔ η_k ↔ S_total` を固定点反復で結合し、体積で重み付けした平均 τ の比（反復後 / ベースライン）で `T_fill` をスケールする。
中央層の温度が固化しきい値を下回ったセルをショートショットとする。
`MultilayerHeleShawSolver(num_layers=5, layer_distribution="wall_refined", thermal_coupling=True)` で呼ぶ。
`num_layers=1` かつ `thermal_coupling=False` なら `HeleShawSolver` と数値的に同一になる（テストで担保）。

`fill_method="march"`（v0.62.0）にすると、固定点の代わりに先端を時間で進める（`core/transient_fill.py`）。満杯のセルの圧力を陰に解き、
先端（p = 0）のセルを流入で陽に満たす。各セルの層は樹脂が着いた時刻で読む（温度の模型は同じで、変わるのは充填の順番の決め方だけ）。
時計は射出率どおりの速度制御。9/14 形状の写真 12 枚では、圧縮前の前半の到達線の RMS が 1.3 mm（固定点は 3.2 mm）になった。

## ライセンス

MIT License — [LICENSE](LICENSE) を参照。

材料パラメータ（`data/materials.json`）は教育目的の generic 値である。実機の検討にはメーカーの実測データを使うこと。
