# 全双工数据集登记与统计（2026-09-01）

## 基本信息

- 处理人：haifengjia（用户本人）
- 数据格式：24 kHz 单声道 WAV + JSONL manifest；时间线 chunk 为 180 ms。
- 分开版：7 份最终版本。
- 合集版：`/nfs/shared_data/duplex_balanced_0818/v3/manifest.jsonl`。
- 机器可读统计：`reports/duplex_dataset_inventory_20260901.stats.json`。
- 合集关系：合集为 7 份分开版的逐行并集，`1,423,071` 条，重复 ID 为 0；合集与分开使用的数据集合一致。
- 正式统计范围不包含带描述数据、外部 GCP/QA_FD 数据、smoke/viewer 测试集。刚生成的 4,203 条本地批次尚未发布到 NFS，也未纳入合集，单列在文末附录。

## 统计口径

- 一条数据是一段单轮或多轮完整对话；轮数取 `source_row.turns` 的元素数。
- query/answer 长度按每一轮分别统计，单位是去除空白后的 Unicode 字符数，标点计入长度。
- answer 为空的静默 `player_complete` 样本按长度 0 计入。
- `TEXT` 表示 timeline 中所有普通文本 token 的出现次数；`<EOR>` 独立统计；每种 FD 控制状态独立统计。
- label 数量是 180 ms 时间线上的 token/状态出现次数，不是样本条数。
- 音频时长来自 manifest 的 `stats.duration_sec`，总时长是训练时间线音频总量。

## 路径与来源

| 数据集 | Manifest | 来源类型 | 数据说明 | 特别注意 |
|---|---|---|---|---|
| AI Partner v6 | /nfs/shared_data/ai_partner_duplex/v6/manifest.jsonl | 内部/已有业务语料处理 | AI 队友角色扮演对话，经范围过滤、场景分配、query TTS、VAD 与全双工时间线构造；做过 F_WAIT 完整性过滤。 | /nfs/shared_data/ai_partner_duplex/v4/wav；v6 仅刷新 manifest，不能删除 v4 WAV。 |
| General RP v5 | /nfs/shared_data/general_rp/v5/manifest.jsonl | 内部通用 RP 语料处理 | 源自 merged_general_rp_sft_data_v2_lf_NTCBT.json 的通用角色扮演对话，经同一全双工流程构造；做过 F_WAIT 完整性过滤。 | /nfs/shared_data/general_rp/v3/wav；v5 复用 v3 WAV。 |
| CPED v5 | /nfs/shared_data/cped_duplex/v5/manifest.jsonl | 开源数据处理 | CPED 对话经场景分配、query TTS、VAD 与全双工时间线构造。官方仓库：https://github.com/scutcyr/CPED；论文：https://arxiv.org/abs/2205.14727。 | 仓库声明 Apache-2.0；WAV 位于 cped_duplex/v1/wav。数据来自中文电视剧，第三方内容权利需另行确认。 |
| CharacterEval v5 | /nfs/shared_data/charactereval_duplex/v5/manifest.jsonl | 开源评测数据处理 | CharacterEval 对话经场景分配、query TTS、VAD 与全双工时间线构造。官方仓库：https://github.com/morecry/CharacterEval；论文：https://arxiv.org/abs/2401.01275。 | 仓库声明 MIT；可能造成 CharacterEval 评测污染。角色源自小说/剧本，档案含百度百科信息，第三方内容权利需另行确认。WAV 位于 v1/wav。 |
| Customized 0806 v6 | /nfs/shared_data/customized_duplex_0806/v6/manifest.jsonl | 模型构造 | 0806 定制化角色扮演文本，经场景分配、F_WAIT 处理、query TTS 和全双工拼接。 | WAV 位于 customized_duplex_0806/v2/wav；最终版本做过已知触顶语音过滤。 |
| Customized 0811+0817 v2 | /nfs/shared_data/customized_duplex_0811_0817/v2/manifest.jsonl | 模型构造 | 0811 与 0817 定制化角色扮演文本合并处理，是合集的主要组成部分。 | 占合集约 75.83%，会显著主导训练分布；WAV 位于 v1/wav；已过滤已知接近 TTS 上限的异常。 |
| Special 0817 v2 | /nfs/shared_data/special_duplex_0817/v2/manifest.jsonl | 模型构造 | 专门构造 ai_intervenes_user 与 player_complete 两类特殊场景。 | intervene 含辱骂、骚扰等敏感文本；complete 的静默停止样本允许 answer 为空。WAV 位于 v1/wav。 |
| 合集 v3 | /nfs/shared_data/duplex_balanced_0818/v3/manifest.jsonl | 上述 7 份的无重复合集 | 逐行合并最终分开版，未重新生成音频；duplicate_ids=0。 | 只有 manifest，音频路径指向上述各数据集的 NFS WAV 目录；迁移时必须整体保留依赖目录。 |

## 规模、时长与轮数

| 数据集 | 条数 | 占合集 | 音频总时长(h) | 单条时长 min/avg/max(s) | 轮数 min/avg/max |
|---|---:|---:|---:|---:|---:|
| AI Partner v6 | 105,477 | 7.41% | 496.5082 | 2.88 / 16.95 / 244.26 | 1 / 1.88 / 9 |
| General RP v5 | 51,168 | 3.60% | 1,121.0547 | 2.88 / 78.87 / 281.88 | 1 / 4.66 / 9 |
| CPED v5 | 7,580 | 0.53% | 39.5538 | 2.70 / 18.79 / 85.32 | 1 / 1.71 / 5 |
| CharacterEval v5 | 856 | 0.06% | 5.0987 | 2.88 / 21.44 / 144.36 | 1 / 1.99 / 9 |
| Customized 0806 v6 | 107,100 | 7.53% | 910.6444 | 3.06 / 30.61 / 237.42 | 1 / 2.99 / 9 |
| Customized 0811+0817 v2 | 1,079,077 | 75.83% | 14,212.6333 | 2.70 / 47.42 / 186.66 | 1 / 3.00 / 6 |
| Special 0817 v2 | 71,813 | 5.05% | 292.4225 | 2.34 / 14.66 / 76.32 | 1 / 1.96 / 3 |
| 合集 v3 | 1,423,071 | 100.00% | 17,077.9155 | 2.34 / 43.20 / 281.88 | 1 / 2.92 / 9 |

## Query 与 Answer 长度

| 数据集 | Query 轮次 | Query 长度 min/avg/max | Answer 轮次 | Answer 长度 min/avg/max |
|---|---:|---:|---:|---:|
| AI Partner v6 | 197,896 | 1.00 / 15.04 / 236.00 | 197,896 | 1.00 / 28.13 / 359.00 |
| General RP v5 | 238,551 | 2.00 / 22.50 / 217.00 | 238,551 | 1.00 / 74.60 / 355.00 |
| CPED v5 | 12,938 | 2.00 / 29.18 / 230.00 | 12,938 | 2.00 / 26.14 / 343.00 |
| CharacterEval v5 | 1,705 | 2.00 / 26.42 / 232.00 | 1,705 | 2.00 / 24.49 / 328.00 |
| Customized 0806 v6 | 319,925 | 2.00 / 16.29 / 81.00 | 319,925 | 2.00 / 41.41 / 251.00 |
| Customized 0811+0817 v2 | 3,240,372 | 2.00 / 22.52 / 158.00 | 3,240,372 | 2.00 / 74.29 / 238.00 |
| Special 0817 v2 | 140,683 | 4.00 / 20.48 / 255.00 | 140,683 | 0.00 / 11.13 / 55.00 |
| 合集 v3 | 4,152,070 | 1.00 / 21.64 / 255.00 | 4,152,070 | 0.00 / 67.26 / 359.00 |

## 场景数量

| 数据集 | normal_qa | interrupts | incomplete | clarification | backchannel | intervene | complete |
|---|---:|---:|---:|---:|---:|---:|---:|
| AI Partner v6 | 70,437 | 13,722 | 9,487 | 898 | 10,933 | 0 | 0 |
| General RP v5 | 32,095 | 8,007 | 4,492 | 1,258 | 5,316 | 0 | 0 |
| CPED v5 | 4,954 | 1,255 | 455 | 85 | 831 | 0 | 0 |
| CharacterEval v5 | 550 | 115 | 51 | 51 | 89 | 0 | 0 |
| Customized 0806 v6 | 70,331 | 17,571 | 7,790 | 46 | 11,362 | 0 | 0 |
| Customized 0811+0817 v2 | 825,806 | 173,822 | 27,134 | 10,062 | 42,253 | 0 | 0 |
| Special 0817 v2 | 0 | 0 | 0 | 0 | 0 | 36,096 | 35,717 |
| 合集 v3 | 1,004,173 | 214,492 | 49,409 | 12,400 | 70,784 | 36,096 | 35,717 |

合集场景占比：

- `normal_qa`：1,004,173（70.564%）
- `player_interrupts_ai`：214,492（15.072%）
- `incomplete_query`：49,409（3.472%）
- `incomplete_query_clarification`：12,400（0.871%）
- `player_backchannel`：70,784（4.974%）
- `ai_intervenes_user`：36,096（2.536%）
- `player_complete`：35,717（2.510%）

## Label 数量

| Label | AI Partner v6 | General RP v5 | CPED v5 | CharacterEval v5 | Customized 0806 v6 | Customized 0811+0817 v2 | Special 0817 v2 | 合集 v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `TEXT` | 3,690,509 | 11,791,642 | 218,312 | 28,576 | 8,269,665 | 150,663,412 | 1,215,650 | 175,877,766 |
| `<EOR>` | 184,174 | 230,544 | 11,683 | 1,590 | 302,354 | 3,066,550 | 129,692 | 3,926,587 |
| `<FD_IDLE>` | 1,881,101 | 3,270,309 | 116,572 | 17,022 | 3,388,735 | 40,950,119 | 1,307,528 | 50,931,386 |
| `<FD_D_WAIT>` | 3,918,680 | 6,859,085 | 427,198 | 52,546 | 5,872,669 | 85,984,373 | 2,994,076 | 106,108,627 |
| `<FD_A_ANSWER>` | 208,829 | 243,867 | 13,769 | 1,794 | 331,287 | 3,282,625 | 129,692 | 4,211,863 |
| `<FD_F_WAIT>` | 10,385 | 5,750 | 540 | 102 | 7,836 | 37,196 | 0 | 61,809 |
| `<FD_G_INTERRUPT>` | 24,655 | 13,323 | 2,086 | 204 | 28,933 | 216,075 | 0 | 285,276 |
| `<FD_H_CONTINUE>` | 10,933 | 5,316 | 831 | 89 | 11,362 | 42,253 | 0 | 70,784 |
| `<FD_J_ACTIVE>` | 898 | 1,258 | 85 | 51 | 46 | 10,062 | 0 | 12,400 |
| `<FD_C_INTERVENE>` | 0 | 0 | 0 | 0 | 0 | 0 | 36,096 | 36,096 |
| `<FD_I_COMPLETE>` | 0 | 0 | 0 | 0 | 0 | 0 | 35,717 | 35,717 |

## CPED 与 CharacterEval 链接及许可证

### CPED

- 官方仓库：https://github.com/scutcyr/CPED
- 许可证原文：https://github.com/scutcyr/CPED/blob/main/LICENSE
- 论文：https://arxiv.org/abs/2205.14727
- 仓库声明许可证：Apache License 2.0。
- 再分发要求要点：附带许可证；修改文件需明确标注发生过修改；保留适用的版权、专利、商标和署名声明；若上游含 NOTICE，还需按 Apache-2.0 要求保留 NOTICE。
- 内容风险：CPED 源自 40 部中文电视剧。Apache-2.0 是仓库顶层许可证，不应直接理解为已清除所有电视剧台词、角色及其他第三方内容的商业使用权。

### CharacterEval

- 官方仓库：https://github.com/morecry/CharacterEval
- 许可证原文：https://github.com/morecry/CharacterEval/blob/main/LICENSE
- 论文：https://arxiv.org/abs/2401.01275
- 仓库声明许可证：MIT License，Copyright (c) 2024 morecry。
- 再分发要求要点：在软件或数据副本及实质性部分中保留原版权声明和 MIT 许可声明。
- 内容风险：官方说明中的角色源自中文小说和剧本，角色档案来自百度百科。MIT 是仓库顶层许可证，不应直接理解为已清除这些第三方原始内容的全部权利。
- 评测风险：CharacterEval 是评测基准；将其转换数据用于训练会造成同名基准的数据污染。

## 使用注意事项

1. 合集 manifest 使用绝对 NFS 音频路径，当前抽查的 7 个数据集音频均可访问；跨机器使用时需要挂载同一 `/nfs/shared_data`。
2. 最终 manifest 会复用旧版本目录中的 WAV。不要只保留最新版本目录，也不要删除上表标明的 v1/v2/v3/v4 WAV 目录。
3. query/player 音频主要由 TTS 合成并使用 ESD 参考音色，不等同于真实用户语音分布；AI 回复主要体现在目标文本与控制标签中。
4. F_WAIT 数据做过“截断前达到 15 字自动过滤 + 模型判断完整性”处理；当时 uncertain 样本采用保留策略。
5. 最新版本已删除已知接近 TTS 生成上限、出现异常拖长的样本，并完成结构校验；这不等同于每条样本都经过人工听检或完整 ASR 一致性校验。
6. CharacterEval 属于评测类来源，训练使用可能污染同名基准；CPED 与 CharacterEval 使用前均应核对上游许可证、用途限制和署名要求。
7. Special 数据中的 intervene 样本有意包含敏感、辱骂或越界文本，用于学习主动制止；下游内容审查和数据分发时要单独标记。
8. Customized 0811+0817 占合集约 75.83%，如果直接训练合集，它会主导角色、文风和长度分布；需要更均衡时应在 sampler 层按数据源或场景重采样。
9. 全量统计解析错误为 0；合集最大 query 长度 255、最大 answer 长度 359，这些是保留的文本长尾，不代表 TTS 触顶异常。

## 附录：尚未纳入合集的本地新批次

- Manifest：`/data/haifengjia/Projects/DuplexAudio/outputs/standard_duplex_48h_20260828/v1/manifest.jsonl`
- 状态：本地可用，尚未复制到 NFS，尚未加入 `duplex_balanced_0818/v3`。
- 来源：角色卡与开场 query 驱动的模型构造数据，不带场景/声音/动作描述。
- 条数：4,203；音频总时长：27.2233 小时；单条时长 min/avg/max：3.96 / 23.32 / 72.36 秒。
- 轮数 min/avg/max：1 / 2.94 / 5。
- Query：12,359 轮，长度 min/avg/max：4 / 15.69 / 93。
- Answer：12,359 轮，长度 min/avg/max：0 / 19.80 / 47。
- 场景：normal 3,127；interrupt 626；incomplete 16；clarification 4；backchannel 209；intervene 110；complete 111。
- Label：TEXT 179,187；EOR 11,694；IDLE 117,513；D_WAIT 222,253；ANSWER 12,529；F_WAIT 20；INTERRUPT 835；CONTINUE 209；J_ACTIVE 4；INTERVENE 110；COMPLETE 111。
- 注意：文本 API 阶段只完成 4,978/200,000 请求，因此不完整 query 数量明显不足；这批不能代表目标配比。

## 一致性校验

- 分开版总行数：1,423,071
- 合集发布行数：1,423,071
- 分开版求和与合集：一致
- 合集重复 ID：0（来自 `release_stats.json`）
- 全量 JSON 解析错误：0
- 合集音频总时长：17,077.9155 小时
