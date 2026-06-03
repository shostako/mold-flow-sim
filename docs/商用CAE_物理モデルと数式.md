# 商用CAEの物理モデルと数式

本資料は、商用CAE（Moldflow / Moldex3D）が射出成形の解析で連成している主な物理モデルと、その代表的な数式をまとめたものです。本ツール（mold-flow-sim）が使うのは、このうち ① の薄肉近似版（Hele-Shaw）と ②（Cross-WLF）の2つだけで、③ 以降は扱いません。商用CAEがどれだけ広い物理を解いているか、そして本ツールが何を割り切ったかを示す対比資料です。

> **注記**: ④〜⑦ はモデル・手法の流派が複数あります（例: 自由表面追跡なら VOF / Control Volume(FAN) / Level-Set、結晶化なら Nakamura / Hoffman-Lauritzen、繊維配向なら Folgar-Tucker とその改良版 ARD-RSC / iARD-RPR）。ここに挙げたのは代表形・骨格で、特に ⑥⑦ の完全形は本来さらに複雑です。

## ① 流動の基礎方程式（保存則）

射出成形の流れを支配する3つの保存則。本ツールはこれを薄肉前提で簡略化した Hele-Shaw 近似だけを使います。

質量保存（連続の式）:

$$\nabla \cdot \mathbf{u} = 0$$

運動量保存（高粘度・慣性を無視した一般化 Stokes 流れ）:

$$\nabla p = \nabla \cdot \left[ \eta \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}} \right) \right]$$

エネルギー保存（移流・拡散＋粘性発熱）:

$$\rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T \right) = k \nabla^2 T + \eta \dot{\gamma}^2$$

## ② Cross-WLF 粘度（本ツールも使用）

温度・剪断速度・圧力に依存する粘度。本ツールも採用する業界標準です。

$$\eta = \frac{\eta_0}{1 + \left(\eta_0 \dot{\gamma}/\tau^*\right)^{1-n}}, \qquad \eta_0(T) = D_1 \exp\!\left[\frac{-A_1(T-T^*)}{A_2 + (T-T^*)}\right]$$

## ③ Tait pvT 状態方程式（2-domain）

樹脂の比容積（密度の逆数）が圧力と温度でどう変わるかを記述し、保圧・収縮の計算に使います。本ツールは保圧を扱わないため未使用。

$$v(T,p) = v_0(T)\left[1 - C\ln\!\left(1 + \frac{p}{B(T)}\right)\right] + v_t(T,p)$$

$$C = 0.0894,\quad v_0(T) = b_1 + b_2(T-b_5),\quad B(T) = b_3\,e^{-b_4(T-b_5)}$$

## ④ 自由表面追跡（VOF / Control Volume 等）

樹脂先端をタイムステップごとに前進させ、充填の進行を直接追います。手法は VOF のほか、中立面解析の Control Volume（FAN: Flow Analysis Network）法など複数あり、下式は代表として VOF を示します。本ツールは時間発展せず一発で解くため未使用。

$$\frac{\partial F}{\partial t} + \mathbf{u} \cdot \nabla F = 0 \qquad (F:\ \text{充填率},\ 0 \le F \le 1)$$

## ⑤ 結晶化（Nakamura モデル）

結晶性樹脂が固まる際の結晶化度の進行を表します。本ツールは未対応。

$$\frac{dX}{dt} = n\,K(T)\,(1-X)\left[-\ln(1-X)\right]^{(n-1)/n}$$

## ⑥ 繊維配向（Folgar-Tucker、骨格）

繊維強化樹脂で、繊維の向き（配向テンソル）の発展を表します。本ツールは未対応。

$$\frac{DA_{ij}}{Dt} = (\text{回転・変形項}) + 2\,C_I\,\dot{\gamma}\,(\delta_{ij} - 3A_{ij})$$

$A_{ij}$＝配向テンソル、$C_I$＝相互作用係数。実際は4次テンソルの縮約を含み、現代の商用CAE実装は改良版（ARD-RSC / iARD-RPR 等）が主流です。

## ⑦ 残留応力・反り（熱粘弾性、骨格）

成形サイクル全体で生じる残留応力を粘弾性まで含めて解き、最終的な反りを予測します。本ツールは未対応。

$$\sigma_{ij}(t) = \int_0^t 2\,G(t-t')\,\frac{d e_{ij}}{dt'}\,dt' + \cdots,\qquad \varepsilon_{\text{thermal}} = \alpha\,\Delta T$$

緩和弾性率 $G$ の畳み込み＋熱収縮から残留応力を出し、構造解析で反りを求めます。

## ⑧ 金型冷却解析（Cool）

金型側の熱マネジメントを解きます。① のエネルギー保存が扱う「樹脂内部の熱」とは別に、金型母材の熱伝導・冷却水路の熱伝達・サイクル平均温度場を求め、冷却時間やサイクルタイム、反り・収縮の起点となる温度の偏りを評価します。本ツールは未対応（壁面冷却モデルは樹脂側の固化であり、金型解析ではありません）。

$$\nabla \cdot \left( k_{\text{mold}} \nabla \bar{T} \right) = 0 \quad (\text{金型内・サイクル平均}), \qquad -k \frac{\partial T}{\partial n} = h\,(T - T_{\text{coolant}}) \quad (\text{冷却水路境界})$$

定常熱伝導（サイクル平均）＋冷却水路の対流境界条件。Moldflow は伝統的に金型側を境界要素法（BEM）で解きます。

## 本ツールとの対応

| 物理モデル | 本ツール |
|----------|---------|
| ① 流動の基礎方程式 | △ 薄肉近似版（Hele-Shaw）のみ |
| ② Cross-WLF 粘度 | ○ 使用 |
| ③ Tait pvT 状態方程式 | × 未対応 |
| ④ 自由表面追跡 | × 未対応（一発解きで代替） |
| ⑤ 結晶化 | × 未対応 |
| ⑥ 繊維配向 | × 未対応 |
| ⑦ 残留応力・反り | × 未対応 |
| ⑧ 金型冷却解析 | × 未対応 |

本ツールが共有するのは ① の簡略版と ② だけで、商用CAEが連成する物理の大半は扱っていません。これは欠落ではなく、薄肉導光板の予備検討に目的を絞った結果です。
