# 全双工数据集可粘贴统计

统计口径：Query/Answer 长度按每轮去除空白后的 Unicode 字符数计算，标点计入；TEXT 合并全部普通文本 token，<EOR> 独立计数，各 FD 状态独立计数。

## AI Partner v6

```text
数据集：AI Partner v6
Manifest：/nfs/shared_data/ai_partner_duplex/v6/manifest.jsonl
总条数：105,477
轮数（最小/最大/平均）：1/9/1.88
Query 长度（最小/最大/平均）：1/236/15.04
Answer 长度（最小/最大/平均）：1/359/28.13
场景数量：normal_qa=70,437；player_interrupts_ai=13,722；incomplete_query=9,487；incomplete_query_clarification=898；player_backchannel=10,933；ai_intervenes_user=0；player_complete=0
Label 数量：TEXT=3,690,509；<EOR>=184,174；<FD_IDLE>=1,881,101；<FD_D_WAIT>=3,918,680；<FD_A_ANSWER>=208,829；<FD_F_WAIT>=10,385；<FD_G_INTERRUPT>=24,655；<FD_H_CONTINUE>=10,933；<FD_J_ACTIVE>=898；<FD_C_INTERVENE>=0；<FD_I_COMPLETE>=0
```

## General RP v5

```text
数据集：General RP v5
Manifest：/nfs/shared_data/general_rp/v5/manifest.jsonl
总条数：51,168
轮数（最小/最大/平均）：1/9/4.66
Query 长度（最小/最大/平均）：2/217/22.50
Answer 长度（最小/最大/平均）：1/355/74.60
场景数量：normal_qa=32,095；player_interrupts_ai=8,007；incomplete_query=4,492；incomplete_query_clarification=1,258；player_backchannel=5,316；ai_intervenes_user=0；player_complete=0
Label 数量：TEXT=11,791,642；<EOR>=230,544；<FD_IDLE>=3,270,309；<FD_D_WAIT>=6,859,085；<FD_A_ANSWER>=243,867；<FD_F_WAIT>=5,750；<FD_G_INTERRUPT>=13,323；<FD_H_CONTINUE>=5,316；<FD_J_ACTIVE>=1,258；<FD_C_INTERVENE>=0；<FD_I_COMPLETE>=0
```

## CPED v5

```text
数据集：CPED v5
Manifest：/nfs/shared_data/cped_duplex/v5/manifest.jsonl
总条数：7,580
轮数（最小/最大/平均）：1/5/1.71
Query 长度（最小/最大/平均）：2/230/29.18
Answer 长度（最小/最大/平均）：2/343/26.14
场景数量：normal_qa=4,954；player_interrupts_ai=1,255；incomplete_query=455；incomplete_query_clarification=85；player_backchannel=831；ai_intervenes_user=0；player_complete=0
Label 数量：TEXT=218,312；<EOR>=11,683；<FD_IDLE>=116,572；<FD_D_WAIT>=427,198；<FD_A_ANSWER>=13,769；<FD_F_WAIT>=540；<FD_G_INTERRUPT>=2,086；<FD_H_CONTINUE>=831；<FD_J_ACTIVE>=85；<FD_C_INTERVENE>=0；<FD_I_COMPLETE>=0
```

## CharacterEval v5

```text
数据集：CharacterEval v5
Manifest：/nfs/shared_data/charactereval_duplex/v5/manifest.jsonl
总条数：856
轮数（最小/最大/平均）：1/9/1.99
Query 长度（最小/最大/平均）：2/232/26.42
Answer 长度（最小/最大/平均）：2/328/24.49
场景数量：normal_qa=550；player_interrupts_ai=115；incomplete_query=51；incomplete_query_clarification=51；player_backchannel=89；ai_intervenes_user=0；player_complete=0
Label 数量：TEXT=28,576；<EOR>=1,590；<FD_IDLE>=17,022；<FD_D_WAIT>=52,546；<FD_A_ANSWER>=1,794；<FD_F_WAIT>=102；<FD_G_INTERRUPT>=204；<FD_H_CONTINUE>=89；<FD_J_ACTIVE>=51；<FD_C_INTERVENE>=0；<FD_I_COMPLETE>=0
```

## Customized 0806 v6

```text
数据集：Customized 0806 v6
Manifest：/nfs/shared_data/customized_duplex_0806/v6/manifest.jsonl
总条数：107,100
轮数（最小/最大/平均）：1/9/2.99
Query 长度（最小/最大/平均）：2/81/16.29
Answer 长度（最小/最大/平均）：2/251/41.41
场景数量：normal_qa=70,331；player_interrupts_ai=17,571；incomplete_query=7,790；incomplete_query_clarification=46；player_backchannel=11,362；ai_intervenes_user=0；player_complete=0
Label 数量：TEXT=8,269,665；<EOR>=302,354；<FD_IDLE>=3,388,735；<FD_D_WAIT>=5,872,669；<FD_A_ANSWER>=331,287；<FD_F_WAIT>=7,836；<FD_G_INTERRUPT>=28,933；<FD_H_CONTINUE>=11,362；<FD_J_ACTIVE>=46；<FD_C_INTERVENE>=0；<FD_I_COMPLETE>=0
```

## Customized 0811+0817 v2

```text
数据集：Customized 0811+0817 v2
Manifest：/nfs/shared_data/customized_duplex_0811_0817/v2/manifest.jsonl
总条数：1,079,077
轮数（最小/最大/平均）：1/6/3.00
Query 长度（最小/最大/平均）：2/158/22.52
Answer 长度（最小/最大/平均）：2/238/74.29
场景数量：normal_qa=825,806；player_interrupts_ai=173,822；incomplete_query=27,134；incomplete_query_clarification=10,062；player_backchannel=42,253；ai_intervenes_user=0；player_complete=0
Label 数量：TEXT=150,663,412；<EOR>=3,066,550；<FD_IDLE>=40,950,119；<FD_D_WAIT>=85,984,373；<FD_A_ANSWER>=3,282,625；<FD_F_WAIT>=37,196；<FD_G_INTERRUPT>=216,075；<FD_H_CONTINUE>=42,253；<FD_J_ACTIVE>=10,062；<FD_C_INTERVENE>=0；<FD_I_COMPLETE>=0
```

## Special 0817 v2

```text
数据集：Special 0817 v2
Manifest：/nfs/shared_data/special_duplex_0817/v2/manifest.jsonl
总条数：71,813
轮数（最小/最大/平均）：1/3/1.96
Query 长度（最小/最大/平均）：4/255/20.48
Answer 长度（最小/最大/平均）：0/55/11.13
场景数量：normal_qa=0；player_interrupts_ai=0；incomplete_query=0；incomplete_query_clarification=0；player_backchannel=0；ai_intervenes_user=36,096；player_complete=35,717
Label 数量：TEXT=1,215,650；<EOR>=129,692；<FD_IDLE>=1,307,528；<FD_D_WAIT>=2,994,076；<FD_A_ANSWER>=129,692；<FD_F_WAIT>=0；<FD_G_INTERRUPT>=0；<FD_H_CONTINUE>=0；<FD_J_ACTIVE>=0；<FD_C_INTERVENE>=36,096；<FD_I_COMPLETE>=35,717
```

## 合集 v3

```text
数据集：合集 v3
Manifest：/nfs/shared_data/duplex_balanced_0818/v3/manifest.jsonl
总条数：1,423,071
轮数（最小/最大/平均）：1/9/2.92
Query 长度（最小/最大/平均）：1/255/21.64
Answer 长度（最小/最大/平均）：0/359/67.26
场景数量：normal_qa=1,004,173；player_interrupts_ai=214,492；incomplete_query=49,409；incomplete_query_clarification=12,400；player_backchannel=70,784；ai_intervenes_user=36,096；player_complete=35,717
Label 数量：TEXT=175,877,766；<EOR>=3,926,587；<FD_IDLE>=50,931,386；<FD_D_WAIT>=106,108,627；<FD_A_ANSWER>=4,211,863；<FD_F_WAIT>=61,809；<FD_G_INTERRUPT>=285,276；<FD_H_CONTINUE>=70,784；<FD_J_ACTIVE>=12,400；<FD_C_INTERVENE>=36,096；<FD_I_COMPLETE>=35,717
```
