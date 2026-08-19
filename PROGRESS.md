# mold-flow-sim 進捗

最終更新: 2026-05-13 14:44 JST

## 現在の状態

- ブランチ: `main` (origin と同期、working tree clean)
- 最新マージ: PR #25 (ICM UI を stroke 一本に簡略化)
- テスト: **165 pass + 1 skip** (`pytest tests/`)
- lint: clean (`ruff check . && ruff format --check .`)
- Streamlit Cloud: <https://mold-flow-sim.streamlit.app> (main 追従、自動再デプロイ)

## 直近の完了 (2026-05-13)

PR #19 〜 PR #25 を 1 日で集中マージ。詳細は `logs/2026-05.md` 参照。

| 機能 | 状態 |
|---|---|
| 圧縮モデル factor → stroke 移行 | ✅ |
| 解析実行前ペインの重複削除 | ✅ |
| 方程式 expander の ICM 拡充 | ✅ |
| 層別 Hele-Shaw ソルバー (PR-A〜F) | ✅ |
| ICM UI を stroke 一本に簡略化 | ✅ |

## 未完了タスク (優先度順)

### gokuusu STEP4 由来 (打合せ後に検討)

| # | タスク | 所要 | メモ |
|---|---|---|---|
| C | **圧力場の陽な出力 + 必要型締力ヘルパー** | 4-5h | バネ計算 7.8t と樹脂圧×投影面積の合算根拠固め。`FlowResult` に圧力場 [Pa] 追加、`required_clamp_force_kN(result, spring_kN=...)` ヘルパー新設。 |
| A | **比較ケース大量生成** | 2-3h | ゲート長 (20mm vs 80mm) × バランサー段数 × 終端肉厚 (A1〜A4) のスイープを `run_demo.py` に追加。`outputs/` に図表ライブラリ化。 |

### 運用改善

| # | タスク | 所要 | メモ |
|---|---|---|---|
| 6 | **tempfile 掃除** | 0.5h | 結果用 tempdir (`app.py` の `tempfile.mkdtemp()`) は session_state 経由で widget 操作後も参照されるので、単純な `with tempfile.TemporaryDirectory()` では壊れる。session cleanup フックか `weakref.finalize` での遅延削除が必要。画像アップロード側の tempfile は v0.23.0 の画像入力撤去で消滅した。 |

### 層別ソルバーの拡張余地 (今回先送り)

- **3D 層境界 surface renderer**: PR-D で先送り。`render_3d_layered_section(result, x_slice_mm)` 等。Plotly Surface trace を N-1 本追加、coloraxis 共有。Plan ファイル `~/.claude/plans/cuddly-coalescing-kazoo.md` に元設計あり。
- **層中央バルク温度の対流項**: 現状は純粋拡散 (1D Neumann)。流れに伴う熱輸送は無視。極厚プレート (h>4mm) や非常に遅い充填で破綻可能。CLAUDE.md に運用範囲を明記済み。
- **局所粘度反復**: V_local の局所評価を `V_avg = injection_velocity_mms` (定数) で代替している。`V_local = (S_total · |∇τ|) / h_total · (1/T_fill)` 形式で導入する余地あり。
- **材料 DB の固化温度フィールド**: 現状は `T_solid = T_mold + 0.3·(T_melt - T_mold)` で派生。材料毎に Tg / 結晶化温度を持たせればショートショット判定の精度向上。

### 長期 (別ロードマップ)

- **完全 3D FVM ソルバー**: 面内コーナー効果・ジェッティング・二次流を真に捕捉するには Hele-Shaw を捨てる必要あり。Stokes 流 + 慣性無視 + 単相で月単位工数、GPU/並列前提。今回は「層別 Hele-Shaw で厚み方向 3D 性のみ」と割り切ってスコープ外。
- **OpenFOAM 連携**: 自前 3D FVM の代替案。geometry を STL / 構造格子で書き出し → 外部 solver → 結果読み込み。学習曲線あり、別検討。
- **結晶化潜熱**: Stefan 問題化。過冷却挙動 (gokuusu STEP3 議論ログ参照) のモデル化に効く。
- **保圧 (パッキング段階)**: 現状は充填までしかモデル化していない。

## 次セッション開始時のアクション

1. **このファイル (PROGRESS.md) を読む** ← 今ここ
2. `git status` で working tree が clean か確認
3. `gh pr list --state open` で未マージ PR がないか確認 (現状はゼロ想定)
4. ユーザーに「次は #C (圧力場 + 型締力)、#A (比較ケース大量生成)、#6 (tempfile)、層別拡張、別件、のどれを優先するか」を確認
5. 必要なら `logs/2026-05.md` を読んで前回の判断履歴を参照

## セッション固有のメモ

- gokuusu STEP4 打合せは 2026-05-13 (本日)。打合せ前の準備は完了状態でクローズ。打合せ後のフィードバック反映は次セッション以降。
- 層別ソルバーの数値挙動 (T_fill_inflation < 1) は物理的に「中央層 ζ=0.5 が高温で η₀ 寄り、ベースライン T_bulk=446K より粘度が低い」現象。実機の固化遅延効果は別途ショートショット判定で扱う設計。
- Streamlit Cloud デプロイは main 追従の自動再デプロイ。ローカル変更を見るには PR を main にマージするか、ローカル `streamlit run app.py` で確認。

## 関連リンク

- 設計プラン: `~/.claude/plans/cuddly-coalescing-kazoo.md` (層別 Hele-Shaw ソルバーの全体設計)
- gokuusu STEP4 議論ログ: `/home/shostako/ClaudeCode/gokuusu/docs/STEP4_ゲート形状検討_議論ログ_2026-05-11.md`
- 作業ログ: `logs/2026-05.md`
- CI: <https://github.com/shostako/mold-flow-sim/actions>
- Streamlit Cloud: <https://mold-flow-sim.streamlit.app>
