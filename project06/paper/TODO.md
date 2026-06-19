# TODO

## 1. コードから確認できなかった実装情報（[要確認]）
- [ ] trainable parameters 数（`model.print_trainable_parameters()` の出力。実ログには未記録）
- [ ] 使用 GPU 名・VRAM（`logs/train_20260611_134432/train.log` に明示なし。`auto_train.sh` は 16GB 前提の分岐あり）
- [ ] effective batch size の最終確認（コード上は batch 8 × grad_accum 4 = 32）
- [ ] 学習時間の厳密値（ログのタイムスタンプ差で約10時間13分と推定）

## 2. 実験実行が必要な評価指標（本文 XXX）
- [ ] Acc@1 / Acc@3 / Acc@5 / Acc@10（`dump_predictions.py` → `evaluate_for_paper.py`）
- [ ] MRR / Exact Match
- [ ] CER / WER
- [ ] KSPC（提案手法は母音キー数/文字数で自動算出。従来手法は実測または定義値）
- [ ] 入力削減率（KSPC 比から算出。基準は従来IME/フリック/QWERTY/かな）
- [ ] 候補選択回数の集計
- [ ] エラー分析 CSV の集計（`errors_proposed.csv`）

## 3. 未実装の比較対象（[今後比較予定]）
- [ ] Base LLM without LoRA の評価（生成型 project06 には未実装）
- [ ] 辞書・頻度ベース baseline（生成型には未実装。reranker系 project02_2 には形式違いで存在）
- [ ] random ranking baseline（未実装）
- [ ] 従来日本語IME/フリック/QWERTY/かな入力の KSPC・画面占有率の実測

## 4. 人間に操作してもらう必要がある評価
- [ ] 入力時間 / 1文字あたり入力時間
- [ ] 誤入力率 / 修正回数 / 候補選択回数 / タスク完了率
- [ ] NASA-TLX / SUS
- [ ] 若年者群 vs 高齢者群
- [ ] フリック熟練者 vs 非熟練者
- [ ] 統計解析（t検定/Wilcoxon、ANOVA/Friedman、効果量、p<.05）

## 5. 参考文献の確認（refs.bib）
- [ ] flick_input / smartwatch_input / ambiguous_input / dasher / gaze_input / vr_input / jpime を実在文献に確定
- [ ] hu2022lora / hart1988nasatlx / brooke1996sus / mackenzie2002kspc / soukoreff2003 の書誌最終確認

## 6. 図表の差し替え
- [ ] fig_topk_accuracy / fig_cer_wer / fig_mrr / fig_kspc / fig_reduction / fig_screen_occupancy / fig_rank_accuracy / fig_error_types を実値で再生成
- [ ] 表 tab:acc, tab:abl, tab:input の XXX を実値に
- [ ] 図中ラベルを日本語化する場合は日本語フォント設定を追加

## 7. Abstract の最終更新
- [ ] 既提出 Abstract と本文の整合確認（「復元」表現の最小化は反映済み）
- [ ] 数値確定後に Abstract に反映するか確認

## 8. 投稿フォーマットへの調整
- [ ] 学会指定スタイルへ documentclass を変更（jarticle → 指定クラス）
- [ ] ページ数 4ページに収まるか確認（表・図の取捨選択）
- [ ] 著者・所属の記入
