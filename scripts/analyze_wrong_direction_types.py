# -*- coding: utf-8 -*-
"""从 eval_detailed.jsonl 统计 direction=wrong 的评判理由错误类型（规则分类）。"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict


def classify_primary(reason: str) -> str:
    """互斥主类：按优先级自上而下匹配第一条。"""
    r = reason

    # 1 误报：标准正常/无异常，模型报异常或虚构
    if re.search(
        r"(正常数据|正常运行状态|全程正常|不存在任何异常|无异常区间|判定全程正常|标准答案.*无异常|数据中不存在).{0,60}"
        r"(识别为异常|指出.*异常|声称.*异常|虚构|捏造|错误地.*异常|存在显著异常|可疑异常)",
        r,
    ):
        return "误报异常（正常判为异常）"
    if re.search(r"错误地将正常|将正常.*识别为异常|根本不存在的.*异常区间|虚构了.*异常", r):
        return "误报异常（正常判为异常）"
    if re.search(
        r"标准答案.*(steady|整体steady|无异常波动|全程无异常).{0,80}(模型.*(存在异常|异常波动|波动较大|剧烈波动))",
        r,
        re.I,
    ):
        return "误报异常（正常判为异常）"

    # 2 漏报：标准有异常，模型否定/说正常/未识别
    if re.search(r"(完全否定|全面否定|否认).{0,25}(存在|有).{0,8}异常", r):
        return "漏报异常（有异常判正常或否认）"
    if re.search(
        r"标准答案.*(存在异常|存在.*异常区间|明确指出).{0,80}"
        r"(模型.*(无异常|没有异常|不存在任何|未识别出|无法判断|声称.*正常|判定.*正常|完全正常))",
        r,
    ):
        return "漏报异常（有异常判正常或否认）"
    if re.search(r"模型(完全|明确).{0,10}判断[^。]{0,20}无异常", r) and re.search(
        r"标准答案.*存在", r
    ):
        return "漏报异常（有异常判正常或否认）"

    # 3 处置/风险方向相反（如无需停机 vs 紧急）
    if re.search(
        r"(无需立即停机|无需停机|severity_factor|轻微).{0,100}(模型.*(紧急|立即|严重风险|绝缘失效))",
        r,
    ) or re.search(r"(模型.*紧急).{0,40}(标准答案.*无需)", r):
        return "处置/风险等级方向相反"
    if re.search(
        r"(错误判断为需立即|应立即停机|立即停机).{0,70}"
        r"(标准答案.*(暂缓|暂无需|无需立即|计划)|与标准答案.*矛盾)",
        r,
    ):
        return "处置/风险等级方向相反"
    if re.search(
        r"(标准答案|标准).{0,50}(紧急降载|暂缓|72小时|计划性停机).{0,80}(模型.{0,30}立即停机)",
        r,
    ):
        return "处置/风险等级方向相反"
    if re.search(
        r"(核心处置方向严重错误|列为最高优先级).{0,40}(立即停机|紧急停机).{0,50}标准答案",
        r,
    ):
        return "处置/风险等级方向相反"

    # 4 核心现象或合理性结论相反
    if re.search(
        r"(合理|不合理).{0,20}(与标准|与标准答案|核心).{0,15}(矛盾|相反)", r
    ) or re.search(r"核心结论.{0,25}(完全相反|矛盾|不一致|直接矛盾)", r):
        return "核心现象/合理性判断相反"

    # 4b 与题干条件或已给定前提矛盾
    if re.search(
        r"(否定|背离|完全否定).{0,15}问题前提|与问题前提.*矛盾|完全否定了问题前提", r,
    ):
        return "与题干条件或前提矛盾"
    if re.search(r"超出正常范围.*正常范围内|问题前提.*矛盾.*标准答案明确指出", r):
        return "与题干条件或前提矛盾"

    # 5 时间步或区间定位错误（judge 明确归入方向错误）
    if re.search(
        r"(时间步|时间区间|异常区间|定位).{0,90}(不重叠|完全不|错误|偏差|超出.*容差|与标准.*不符|严重偏离)",
        r,
    ):
        return "时间步/区间定位错误"
    if re.search(r"(错误定位|定位在).{0,50}(与标准|标准答案).{0,20}(不一|偏离|错位|矛盾)", r):
        return "时间步/区间定位错误"
    if re.search(r"(时间区间|异常区间|起止|范围).{0,40}无重叠|无重叠.{0,25}(标准答案|与标准)", r):
        return "时间步/区间定位错误"
    if re.search(r"(不重叠|超出.*容差).{0,40}方向", r):
        return "时间步/区间定位错误"

    # 6 机理、归因、故障类型与标准矛盾（含 judge 常用句式；放宽窗口避免“偏离”出现在句末时漏判）
    if re.search(
        r"(物理机制|因果链|归因|归因为|根因|故障类型|根本原因|传播路径|因果机制).{0,55}"
        r"(错误|偏离|矛盾|不符|本质偏差|根本矛盾|完全无关|臆测)",
        r,
    ):
        return "机理/归因/故障类型错误"
    if re.search(
        r"将[^。]{0,55}(错误归|归因于).{0,180}(矛盾|相反|偏离|不符|本质|标准答案明确|物理本质|完全偏离)", r,
    ):
        return "机理/归因/故障类型错误"
    if re.search(r"(将根本原因|根本原因).{0,20}错误判定|因果.{0,15}(颠倒|相反)|混淆因果方向", r):
        return "机理/归因/故障类型错误"
    if re.search(
        r"(故障机制|事件类型|物理本质|工况).{0,30}(相反|矛盾|对立|不符)|物理现象.{0,20}(升温|降温).{0,15}(降温|升温)",
        r,
    ):
        return "机理/归因/故障类型错误"
    if re.search(
        r"(提出的|猜测|归结).{0,25}与标准答案.{0,40}(无关|机理完全无关|控制工程)", r,
    ):
        return "机理/归因/故障类型错误"

    # 7 运行模式 / 事件形态误判（steady、启停、瞬时 vs 持续等）
    if re.search(
        r"(steady|启停|停机周期|受控停机|计划检修).{0,50}(矛盾|相反|误判)|"
        r"(运行模式|全程无异常|始终正常).{0,40}(虚构|误判为|错误地).{0,20}停机",
        r,
        re.I,
    ):
        return "运行模式/事件形态误判"
    if re.search(r"(瞬时|突发).{0,30}(持续|渐进|缓慢)|持续性.{0,20}瞬时", r):
        return "运行模式/事件形态误判"

    # 8 传感器趋势或物理量升降方向与标准相反
    if re.search(
        r"(上升|升高|增大).{0,25}(标准答案|标准).{0,25}(下降|降低|减小)|"
        r"(下降|降低|减小).{0,25}(标准答案|标准).{0,25}(上升|升高|增大)",
        r,
    ):
        return "趋势/升降方向与标准相反"
    if re.search(r"降温.*升温|升温.*降温|均值下降.*上升|下降趋势.*误判为.*上升", r):
        return "趋势/升降方向与标准相反"

    # 9 标准关键信息未识别或严重遗漏（有输出但不覆盖标准中的核心判据/区间/机理）
    if re.search(
        r"(完全未识别|未能识别出|未识别出标准答案中|完全忽略|完全未提及标准答案|严重遗漏).{0,120}"
        r"(异常|特征|机理|判据|事件|区间|原因|依据|松动|放电|泄漏|卡滞|过热|前提|核心)",
        r,
    ):
        return "标准关键信息未识别或严重遗漏"
    if re.search(
        r"模型未识别出.{0,90}(关键|核心).{0,35}(异常|故障|特征|机理|事件|区间)", r,
    ):
        return "标准关键信息未识别或严重遗漏"
    if re.search(r"(未进行任何定量|未引用.*关键阈值|未利用.{0,20}排除性证据).{0,80}(标准答案|矛盾|相悖)", r):
        return "标准关键信息未识别或严重遗漏"

    # 10 分析关注点或故障部位错位（抓错主因/主设备）
    if re.search(
        r"(错误地将|反而将|错误聚焦).{0,60}(作为|列为|聚焦|关注).{0,25}(首要|核心|主要).{0,50}标准答案明确", r,
    ):
        return "分析关注点或故障部位错位"

    # 11 数值、统计量或取值范围严重误读
    if re.search(
        r"(误读为|错标为|误判为).{0,55}\d|数值.{0,30}(严重|大幅).{0,25}(矛盾|错误|不符|失真)|相差.{0,10}倍|"
        r"数量级|统计特征.{0,15}(严重不符|不符)|全部数值.*失真|范围错标|夸大.{0,8}倍",
        r,
    ):
        return "数值或统计量严重误读"

    # 12 通道、命名、参数或量纲混淆
    if re.search(
        r"(误称为|误标为|混淆).{0,45}(通道|命名|参数|物理量|量纲)|量纲混乱|重复描述|非标准命名", r,
    ):
        return "通道命名或术语混淆"

    # 13 因果链条或系统耦合曲解（含“无直接关系却强加因果”）
    if re.search(
        r"(无直接因果|不应视.*因果|因果方向|因果链).{0,70}(矛盾|相反|相悖|颠倒|忽略)", r,
    ):
        return "因果链条或系统耦合曲解"
    if re.search(
        r"标准答案明确指出.{0,40}(无直接|并行响应|非因果).{0,60}(模型|反而|错误)", r,
    ):
        return "因果链条或系统耦合曲解"

    # 14 多段碎片化异常或次数误判（标准单一连续段）
    if re.search(r"(判断|认为|指出).{0,12}存在.{0,12}\d+次异常", r):
        return "多段碎片化异常误判"
    if re.search(r"两次停机|多处.{0,20}异常区间", r) and re.search(
        r"(标准答案|标准).{0,40}(单一|唯一|连续|整体)", r,
    ):
        return "多段碎片化异常误判"

    # 15 否认标准已给定前提或拒绝确认
    if re.search(
        r"(根本否认|完全否认).{0,20}(存在|有).{0,12}(异常|故障|事件|放电|偏航)", r,
    ):
        return "否认标准前提或拒绝确认"
    if re.search(r"(无法确认|声称无法).{0,40}标准答案明确", r):
        return "否认标准前提或拒绝确认"
    if re.search(r"质疑.{0,25}数据.{0,15}有效.{0,60}标准答案", r):
        return "否认标准前提或拒绝确认"

    # 16 异常性质或风险后果定性错误（如提升/稳定 vs 退化、中等 vs 轻微）
    if re.search(r"颠倒了异常性质", r):
        return "异常性质或后果定性错误"
    if re.search(r"(错误解读为|误判为).{0,25}(性能提升|高功率运行模式|稳定且无异常)", r) and re.search(
        r"(标准答案|退化|隐性|异常|矛盾)", r,
    ):
        return "异常性质或后果定性错误"
    if re.search(
        r"(风险等级|后果判断).{0,25}(标准答案|相反|矛盾)|错误断定.{0,30}严重", r,
    ):
        return "异常性质或后果定性错误"

    # 17 答非所问或未覆盖任务要求（措施、停机决策等）
    if re.search(
        r"未回答问题|完全偏离问题要求|完全偏离问题|仅孤立|未涉及.*关键|"
        r"未回答.{0,10}核心问题|未给出任何.{0,30}(运维|措施|建议|处置)|"
        r"完全未提及.{0,25}(停机|处置|检修|优先级)|泛泛而谈|未按.*要求|"
        r"回答中断|截断在|回答未完成|仅输出了|未提供任何具体措施|回答完全缺失|中途截断",
        r,
    ):
        return "答非所问/未覆盖核心"

    # 17b 全局/周期性运行模式或多变量协同特征遗漏
    if re.search(
        r"(仅聚焦|仅关注).{0,55}(单点|局部).{0,45}(忽略|未识别).{0,25}(全局|整体|全程|周期性|多变量协同)", r,
    ):
        return "全局或协同模式识别遗漏"
    if re.search(
        r"忽略.{0,20}(全局性|多变量协同|周期性启停).{0,25}特征", r,
    ):
        return "全局或协同模式识别遗漏"

    # 17c 多通道“同步异常”误判（与标准“无同步/单一通道”矛盾）
    if re.search(
        r"(多个|各)通道.{0,35}(均|同步).{0,25}(异常|判定).{0,70}(其余通道|无同步|关键结论矛盾)",
        r,
    ):
        return "多通道协同性误判"

    # 17d 事实陈述与标准数据严重不符（综合表述）
    if re.search(
        r"(事实性错误|事实基础矛盾|严重违背标准答案.{0,20}(实际|给出|范围)|数值.{0,15}荒谬|全面错误)", r,
    ):
        return "事实陈述与标准严重不符"

    # 17e 问题聚焦点或分析对象错位（子系统、主题偏离）
    if re.search(
        r"(分析重点|将分析重点|错误地将分析).{0,35}放在.{0,70}(无关|次要).{0,30}标准答案|"
        r"完全偏离了标准答案.{0,25}所聚焦|未识别核心问题.{0,40}(反而|错误聚焦)",
        r,
    ):
        return "问题聚焦点或分析对象错位"

    # 17f 时间尺度、周期长度或单位表述错误
    if re.search(
        r"(时间尺度|监测周期|周期长度).{0,20}(错误|失真|错位|误|严重)|"
        r"(单位|量纲).{0,12}错误.{0,25}(应为|标准答案)|将单位错误",
        r,
    ):
        return "时间尺度或单位表述错误"

    # 17g 异常通道/变量角色指认错误（认错哪条通道为异常主体）
    if re.search(
        r"标准答案.{0,60}(异常通道|关键通道).{0,40}(是|为).{0,40}(而模型|模型).{0,25}(错误)", r,
    ):
        return "异常通道或变量角色指认错误"

    # 17h 稳态/小幅变化 vs 剧烈波动判定相反
    if re.search(
        r"标准答案.{0,50}(基本不变|保持稳定|无异常波动|steady).{0,55}"
        r"(模型.{0,25}(剧烈|大幅|并非保持|波动较大))|"
        r"标准答案.{0,50}负载.{0,15}稳定.{0,40}(模型.{0,20}剧烈波动)",
        r,
        re.I,
    ):
        return "波动幅度或稳定性判定相反"

    # 17i 日周期、季节性或长周期规律遗漏
    if re.search(
        r"未识别.{0,30}(日周期|季节性|长周期).{0,25}(波动|规律|特征|模式)", r,
    ):
        return "长周期/外部规律特征遗漏"

    # 17j 处置步骤优先级或序列与标准应急逻辑相反
    if re.search(
        r"(列为第一优先级|第一优先级).{0,25}(立即停机|紧急).{0,60}"
        r"标准答案.{0,40}(先降负荷|降载|再冷却|核心处置逻辑)", r,
    ):
        return "处置步骤顺序或优先级结构错误"

    # 18 泛化：评判明确写出与标准核心结论相反，但前面规则未命中
    if re.search(
        r"(方向完全相反|根本矛盾|完全矛盾|核心结论.{0,20}相反|核心事实判断.{0,15}矛盾|方向性错误|"
        r"内容与标准答案.{0,20}全面偏离|专业性和方向性严重偏离)",
        r,
    ):
        return "与标准核心结论相反（泛化）"

    return "多种原因混合"


def classify_other_detail(reason: str) -> str:
    """
    对主类「多种原因混合」的 reason 进行二次细分，便于报告展示。
    子类互斥，按顺序优先匹配；仍为规则归纳，非人工标注。
    """
    r = reason

    if re.search(
        r"(物理约束|工程边界|超出.*上限|违背.{0,8}物理|不可能同时).{0,40}(模型|声称|描述)", r,
    ):
        return "其他子类-物理约束或工程可行边界理解错误"
    if re.search(
        r"(虚构|捏造).{0,15}(桨|角度|RPM|转速).{0,15}(标准答案|实际|范围)", r,
    ):
        return "其他子类-物理约束或工程可行边界理解错误"

    if re.search(
        r"(不应优先|次要.*列为|第二优先级).{0,40}(标准答案|唯一|首要|第一优先级)|"
        r"未识别.{0,15}(首要|最高优先级).{0,25}(处置|措施|检修)",
        r,
    ):
        return "其他子类-处置层级或证据优先级理解错误"

    if re.search(
        r"(测点|通道).{0,15}(主次|关系).{0,25}(与标准答案矛盾|标准答案明确)|"
        r"winding_temp_[12].{0,30}(与标准答案|标准答案).{0,20}(矛盾|更高|更低)",
        r,
    ):
        return "其他子类-测点关系或对比结论错误"

    if re.search(
        r"时间步.{0,20}(误|错误).{0,15}(小时|分钟|天)|将.{0,12}\d+.{0,15}步.{0,15}(等同|视为).{0,10}(小时|分钟)",
        r,
    ):
        return "其他子类-步长与时间换算理解错误"
    if re.search(
        r"关注了.{0,45}(无关|次要).{0,25}通道.{0,35}遗漏.{0,20}(关键|核心|标准答案)", r,
    ):
        return "其他子类-时空与关键判据复合错误"

    if re.search(
        r"(上升|下降|波动|稳定|斜率|趋势).{0,55}(标准答案|标准).{0,35}(矛盾|相反|不符)", r,
    ):
        return "其他子类-趋势或统计口径与标准矛盾"
    if re.search(
        r"(标准答案).{0,45}(上升|下降|稳态).{0,35}(模型.{0,25}(相反|误判|矛盾))",
        r,
    ):
        return "其他子类-趋势或统计口径与标准矛盾"

    if re.search(
        r"(将故障|将异常|根本原因|核心异常).{0,40}(归|定性|判定为).{0,70}(标准答案明确指向|而标准答案)", r,
    ):
        return "其他子类-根因对象或故障模式指认偏差"
    if re.search(r"物理机制.{0,35}根本不同|机理判断完全相反|核心故障机理判断完全相反", r):
        return "其他子类-根因对象或故障模式指认偏差"

    if re.search(
        r"(虽有|部分).{0,25}(细节|分析|合理).{0,35}(但|然而).{0,20}(方向|结论|核心).{0,15}(错误|相反)",
        r,
    ):
        return "其他子类-局部合理但整体结论相反"

    if re.search(
        r"(逻辑断裂|论证跳跃|严重不完整|中断于|未得出.{0,8}结论|属严重不完整)", r,
    ):
        return "其他子类-论证不完整或输出截断"

    if re.search(
        r"(罗列了|列举).{0,12}\d{0,2}.{0,8}(种|类|项).{0,15}(可能|猜测|因素).{0,45}(缺乏证据|空泛|未利用)",
        r,
    ):
        return "其他子类-多假设罗列缺证据"

    if re.search(
        r"(强烈建议|需高度关注).{0,35}(标准答案.{0,30}(无需|正常振荡|无参数超))|"
        r"(标准答案.{0,35}无需).{0,50}(模型.{0,25}(立即停机|强烈))",
        r,
    ):
        return "其他子类-风险急迫性整体判断反号"

    if re.search(
        r"解读与事实相反|与事实相反|依据与标准答案矛盾|整体未进行.*量化", r,
    ):
        return "其他子类-事实解读或量化支撑不足"

    return "其他子类-混合或低频句式"


def classify_other_mix_opening(reason: str) -> str:
    """对「其他子类-混合或低频句式」按句首粗分桶，仅用于补充统计。"""
    s = reason.strip()
    prefixes = [
        ("模型完全", "混合句首-模型完全…"),
        ("模型错误", "混合句首-模型错误…"),
        ("模型未", "混合句首-模型未…"),
        ("模型将", "混合句首-模型将…"),
        ("模型回答", "混合句首-模型回答…"),
        ("模型对", "混合句首-模型对…"),
        ("核心", "混合句首-核心…"),
        ("方向错误", "混合句首-方向错误…"),
    ]
    for pref, label in prefixes:
        if s.startswith(pref):
            return label
    if s.startswith("模型"):
        return "混合句首-模型…(其他接续)"
    return "混合句首-非上述模式"


def multi_tag(reason: str) -> set[str]:
    """多标签辅助：用于观察共现。"""
    r = reason
    tags: set[str] = set()
    if re.search(r"正常.*异常|将正常|虚构|捏造|不存在任何异常", r):
        tags.add("含:误报线索")
    if re.search(r"否定|否认|无异常|没有异常|未识别.*异常", r):
        tags.add("含:漏报线索")
    if re.search(r"时间步|区间|定位|不重叠|容差", r):
        tags.add("含:时空定位线索")
    if re.search(r"机理|归因|因果|物理", r):
        tags.add("含:机理解释线索")
    if re.search(r"矛盾|相反", r):
        tags.add("含:矛盾表述")
    return tags


def load_wrong_samples(eval_detailed_path: str) -> list[dict]:
    wrongs: list[dict] = []
    with open(eval_detailed_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if s.get("eval", {}).get("direction") == "wrong":
                wrongs.append(s)
    return wrongs


def compute_stats_from_wrongs(wrongs: list[dict]) -> dict:
    n = len(wrongs)
    ctr = Counter()
    by_domain: dict[str, Counter] = defaultdict(Counter)
    by_level: dict[str, Counter] = defaultdict(Counter)
    by_diff: dict[str, Counter] = defaultdict(Counter)
    tag_ctr: Counter[str] = Counter()

    other_bucket = "多种原因混合"
    other_sub: Counter[str] = Counter()
    mix_opening: Counter[str] = Counter()

    for s in wrongs:
        reason = s["eval"]["reason"]
        cat = classify_primary(reason)
        ctr[cat] += 1
        by_domain[s.get("domain", "?")][cat] += 1
        by_level[s.get("level", "?")][cat] += 1
        by_diff[s.get("difficulty", "?")][cat] += 1
        for t in multi_tag(reason):
            tag_ctr[t] += 1
        if cat == other_bucket:
            od = classify_other_detail(reason)
            other_sub[od] += 1
            if od == "其他子类-混合或低频句式":
                mix_opening[classify_other_mix_opening(reason)] += 1

    other_n = int(ctr[other_bucket])

    out_d: dict = {
        "wrong_n": n,
        "primary_type_counts": dict(ctr),
        "primary_type_ratio": {k: round(v / n, 4) for k, v in ctr.items()} if n else {},
        "by_domain": {d: dict(c) for d, c in sorted(by_domain.items())},
        "by_level": {lv: dict(c) for lv, c in sorted(by_level.items())},
        "by_difficulty": {d: dict(c) for d, c in sorted(by_diff.items())},
        "multi_tag_counts": dict(tag_ctr),
    }
    if other_n:
        out_d["other_n"] = other_n
        out_d["other_subtype_counts"] = dict(other_sub)
        out_d["other_subtype_ratio"] = {k: round(v / other_n, 4) for k, v in other_sub.items()}
        mix_n = int(other_sub.get("其他子类-混合或低频句式", 0))
        if mix_n:
            out_d["other_mix_opening_counts"] = dict(mix_opening)
            out_d["other_mix_opening_ratio"] = {
                k: round(v / mix_n, 4) for k, v in mix_opening.items()
            }
    else:
        out_d["other_n"] = 0
        out_d["other_subtype_counts"] = {}
        out_d["other_subtype_ratio"] = {}

    return out_d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "eval_detailed",
        help="eval_detailed.jsonl 路径",
    )
    ap.add_argument(
        "--json-out",
        default="",
        help="可选：写出聚合 JSON",
    )
    args = ap.parse_args()

    wrongs = load_wrong_samples(args.eval_detailed)
    n = len(wrongs)
    out = compute_stats_from_wrongs(wrongs)
    out["source"] = args.eval_detailed

    ctr = Counter(out["primary_type_counts"])
    tag_ctr = Counter(out["multi_tag_counts"])

    lines = []
    lines.append(f"方向错误样本数 N={n}")
    lines.append("主类分布（互斥，按规则优先级）:")
    for k, v in ctr.most_common():
        lines.append(f"  {v:5d}  {100.0 * v / n:5.1f}%  {k}")

    lines.append("\n多标签线索占比（可重叠，仅作辅助）:")
    for k, v in tag_ctr.most_common():
        lines.append(f"  {v:5d}  {100.0 * v / n:5.1f}%  {k}")

    osub = out.get("other_subtype_counts") or {}
    if osub:
        on = out.get("other_n", 0) or 1
        lines.append(f'\n「多种原因混合」二次细分（仅针对该主类，N={on}）:')
        for k, v in sorted(osub.items(), key=lambda x: -x[1]):
            lines.append(f"  {v:5d}  {100.0 * v / on:5.1f}%  {k}")
        om = out.get("other_mix_opening_counts") or {}
        if om:
            mn = int(osub.get("其他子类-混合或低频句式", 0)) or 1
            lines.append(
                f'\n「多种原因混合」中混合句式句首粗分桶（N={mn}，占该主类 {100*mn/on:.1f}%）:'
            )
            for k, v in sorted(om.items(), key=lambda x: -x[1]):
                lines.append(f"  {v:5d}  {100.0 * v / mn:5.1f}%  {k}")

    report = "\n".join(lines)
    print(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.json_out}")


if __name__ == "__main__":
    main()
