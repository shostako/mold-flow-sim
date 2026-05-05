# mold-flow-sim

射出成形の流動解析を Hele-Shaw 近似 + Cross-WLF 粘度モデル + Pseudo-Conduction 法で
**極端に簡略化した** Python シミュレータ。教育・初期スクリーニング・概念検証用。

> ⚠ **これは Moldflow / Moldex3D の代替ではない。** 過渡熱結合・3D 流れ・固化層・
> 結晶化・収縮反り・保圧 — どれもまだモデル化していない。実機の型設計検討は商用 CAE を使え。

## 何ができるか（現状）

- 2D 構造格子上で薄肉キャビティの**疑似充填時間場** τ を一発解（楕円型方程式 ∇·(S∇τ)=1）
- 充填先端アニメーション（GIF）、相対圧力マップ、ウェルドライン候補、エアトラップ候補
- パラメータ：樹脂（PP / ABS / PC / PA66 / PMMA）、樹脂温度、金型温度、射出速度、射出体積流量
- 形状入力：合成プレート + ランナー + スプルー、または閾値処理した PNG/JPG 画像
- 簡易圧縮成形 (ICM) 等価モデル

## 何ができないか（既知の制限）

| 項目 | 状態 |
|------|------|
| 過渡熱結合（壁面冷却・固化層） | ❌ 未実装 |
| 真の 3D 流れ・ジェッティング | ❌ 未実装 |
| 結晶化・収縮反り | ❌ 未実装 |
| パッキング段階の保圧 | ❌ 未実装 |
| 局所せん断速度反復 | ❌ 単一代表値のみ |
| STL / STEP 直接読込 | ❌ 画像のみ |
| 中立面メッシュ（非構造格子） | ❌ 構造格子のみ |
| 解析解検証テスト | ⚠ 整備中 |

## ロードマップ（射程：初期スクリーニングツール）

1. **基盤整備**（パッケージ化、CI、テスト、検証）
2. **形状入力強化**：パラメトリックなフィルムゲート / サイドゲートランナー、中立面メッシュ
3. **物理モデル拡充**：局所粘度反復、簡易過渡熱、固化層厚 → 実効ギャップ縮小
4. **数値の地盤強化**：行列組立ベクトル化、反復法 (CG / AMG)、メッシュ収束テスト
5. **入出力実用化**：VTK エクスポート、材料 DB の出典明記・拡張
6. **ICM の正直化**：充填フェーズ + 圧縮フェーズの 2 相モデル

## インストール

```bash
git clone https://github.com/shostako/mold-flow-sim.git
cd mold-flow-sim
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # ランタイムのみ
pip install -e ".[dev]"              # 開発用（ruff, pytest 含む）
```

Python 3.11 以上必須。

## 実行

### Streamlit UI（インタラクティブ）

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開く。サイドバーで形状・材料・射出条件を設定し
「解析実行」ボタンで結果（GIF・圧力マップ・ウェルドライン）を取得。

### CLI バッチ（パラメータスイープ）

```bash
python run_demo.py
python run_demo.py --cases PP_baseline PP_dual_gate
```

`outputs/<case>/` に GIF・PNG・連番フレームを出力。

## 物理モデル概要

### Hele-Shaw 近似 + Pseudo-Conduction 法

充填過程を時間進行ではなく、楕円型問題の一発解で解く：

```
∇·(S ∇τ) = 1     (キャビティ内)
τ = 0            (ゲート, Dirichlet)
S∇τ·n = 0        (壁面, Neumann)
S = h³ / (12·η_eff)
```

- `τ` は擬似到達時間場（ゲートからの "距離" の単調関数）
- 絶対時間スケーリング：`fill_time(x,y) = (τ/τ_max) × (V_cavity / Q)`
- 面コンダクタンスは隣接セルの調和平均

### Cross-WLF 粘度モデル

```
η(γ̇, T, P) = η₀(T,P) / (1 + (η₀ γ̇ / τ*)^(1−n))
η₀(T,P) = D₁ · exp(−A₁ (T−T*) / (Ã₂ + (T−T*)))
T* = D₂ + D₃ P
```

材料パラメータは `data/materials.json`（generic 値）。

## ライセンス

MIT License — [LICENSE](LICENSE) を参照。

材料パラメータ（`data/materials.json`）は教育目的の generic 値であり、
実機検討にはベンダー実測データを使うこと。
