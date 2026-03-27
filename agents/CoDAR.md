# CoDAR

数据input结构：

```
SCENARIOS = ["affection", "attitude", "intent"]
VALID_MECHANISMS = {
    "affection": ["multimodal incongruity", "figurative semantics", "affective deception", "socio_cultural dependency"],
    "intent": ["prosocial deception", "malicious manipulation", "expressive aggression", "benevolent provocation"],
    "attitude": ["dominant affiliation", "dominant detachment", "protective distancing", "submissive alignment"]
}
VALID_LABELS = {
    "affection": ["happy", "sad", "disgusted", "angry", "fearful", "bad"],
    "attitude": ["supportive", "appreciative", "sympathetic", "neutral",
                 "indifferent", "concerned", "skeptical", "dismissive", "disapproving", "contemptuous", "hostile"]
    "intent": ["mitigate", "intimidate", "alienate", "mock", "denounce", "provoke", "dominate", "condemn"]
}
```

- Affection下各部分定义 :

```python
1. "mechanism": 
- "multimodal incongruity": Implicit affection arises from polarity conflict or mutual exclusion between modalities (text vs image, or image-context vs text), such that the literal meaning is negated or reframed by the other modality. Key signature: "What is said" and "what is shown" cannot both be true in the same frame -> the affective state is inferred from the conflict.
- "figurative semantics": The affective state is conveyed via source->target conceptual mapping rather than direct emotion words or standard displays. Key signature: The sample "talks about X" but means an affective state through metaphor, symbol, hyperbole, understatement, or poetic imagery.
- "affective deception": The affective state is deliberately masked (performed calmness/harshness), but involuntary cues "leak" the underlying affect. Key signature: "Displayed affect" != "true affective state" inferred from leakage.
- "socio_cultural context dependency": The affective state can only be interpreted correctly using external world knowledge (memes, events, cultural codes, relationship norms). Key signature: The pair is semantically opaque without a shared cultural reference that encodes affect indirectly.

2. "label": 
- happy
  * Level-2 Sub-labels: Playful, Content, Interested, Proud, Accepted, Powerful, Peaceful, Trusting, Optimistic.
  * Core Mechanism: A positive state involving satisfaction, joy, or confidence, usually accompanied by positive evaluation of the current scenario (e.g., feeling accepted or hopeful).
  * Boundary Rule: If the expression is primarily factual with extremely weak emotion and lacks clear positive cues, classify as 'Bad' as the low-intensity fallback required by this label set. If the core mechanism is gratitude/thankfulness for someone's help, it strictly falls under 'Happy'.
- sad
  * Level-2 Sub-labels: Lonely, Vulnerable, Despair, Guilty, Depressed, Hurt.
  * Core Mechanism: A negative state associated with loss, helplessness, or relationship setbacks, often leading to withdrawal or lack of motivation.
  * Boundary Rule: If the focus is on self-blame/compensation for a mistake, map to the 'Guilty' sub-label. If the focus is relational pain caused by others, map to the 'Hurt' sub-label. 
- disgusted
  * Level-2 Sub-labels: Repelled, Awful, Disappointed, Disapproving.
  * Core Mechanism: A strong aversion or rejection toward physical stimuli (smell/food) or social/moral stimuli (immorality, hypocrisy, offense).
  * Boundary Rule: The core tendency is "AWAY FROM" (want to avoid/reject/deny). If the emotion primarily involves blame, confrontation, or emphasizing "the other party is wrong" rather than mere aversion, it leans towards 'Angry'.
- angry
  * Level-2 Sub-labels: Let down, Humiliated, Bitter, Mad, Aggressive, Frustrated, Distant, Critical.
  * Core Mechanism: Triggered by being offended, hindered, or treated unfairly, leading to hostility, frustration, and an intention to control, fight back, or argue.
  * Boundary Rule: The core tendency is "AGAINST". If the expression is merely cold or "I don't care" (without confrontation), it leans towards 'Bad'. If it's primarily aversion without the urge to fight back, it leans towards 'Disgusted'.
- fearful
  * Level-2 Sub-labels: Scared, Anxious, Insecure, Weak, Rejected, Threatened.
  * Core Mechanism: Centered around threat and insecurity, expecting that "something bad might happen / I might get hurt," leading to a tendency to avoid, seek protection, or increase vigilance.
  * Boundary Rule: Includes continuous worry caused by long-term uncertainty ('Anxious'). If it's just a brief reaction to unexpected factual info without a core sense of threat, map it to 'Bad' based on context.
- bad
  * Level-2 Sub-labels: Bored, Busy, Stressed, Tired.
  * Core Mechanism: A generalized negative state for scenarios that do not fit the specific intensity or clear triggers of Sad, Angry, Disgusted, or Fearful. It includes feeling uncomfortable, exhausted, apathetic, or experiencing cognitive dissonance.
  * Boundary Rule: Use this category for low-intensity negative states or mixed/cynical contexts. For instance, implicit scenarios involving "dark humor" should be classified here. The mechanism of dark humor relies on taboo, tragedy, or moral discomfort, which aligns with this generalized uncomfortable/apathetic state, rather than genuine positive joy (Happy) or targeted hostility (Angry).
```

- Attitude 下各部分定义 :

```python
1. "mechanism": 
- "dominant affiliation": Surface friendliness or acceptance is used to assert a "stronger to weaker" superiority; closeness is granted from above. Key signatures include: Talking Down (treating the target like a child to establish superiority), Patronizing Praise (praising while implicitly lowering the standard for the target), or Benevolent Control (depriving the target's autonomy under the guise of "for your own good").
- "dominant detachment": Establishing high status while cutting off emotional connection. Key signatures include: Invalidation (standing on a moral/intellectual high ground to deny the target's value or feelings), Character Blaming (attributing the target's plight to their own moral/personality flaws; "they asked for it"), or Rationalizing Over (using cold logic to dismiss the target's perspective as unworthy of serious attention).
- "protective distancing": A lower-power or vulnerable stance avoiding direct confrontation via non-commitment or psychological isolation. Key signatures include: Keeping It Open (suspending commitment/action without explicitly agreeing or disagreeing), Emotional Withdrawal (receiving information but noticeably dropping emotional engagement), or Skeptical Distance (appearing open but implicitly refusing to accept the target's premises).
- "submissive alignment": Proactively lowering one's own status to seek safety, attachment, or protection. Key signatures include: Self-Diminishment (belittling oneself to gain acceptance), Over-Accommodation (sacrificing one's own stance to avoid conflict/rejection), or Leaning On (showing weakness to signal a need for the target's protection or decision-making).

2. "label": 
- "supportive"
  * Core Definition: Explicitly defends the target or affirms their legitimacy and value.
  * Anchor Examples (Non-exhaustive): e.g., making excuses for their failure, protecting their reputation, taking their side.
  * Boundary Rule: The focus is on active alignment/defense. If it only evaluates output positively without defending or taking a side, map to 'Appreciative'.
- "appreciative"
  * Core Definition: Positive evaluation of the target's abilities, qualities, or achievements.
  * Anchor Examples (Non-exhaustive): e.g., praising effort, complimenting results or design.
  * Boundary Rule: The focus is on merit evaluation. It differs from 'Supportive' as it doesn't necessarily involve defending the target against adversity.
- "sympathetic"
  * Core Definition: Empathy and understanding for the target's unfavorable scenario.
  * Anchor Examples (Non-exhaustive): e.g., emphasizing harsh environments, bad luck, or high difficulty to comfort and downplay responsibility.
  * Boundary Rule: The focus is on shared suffering/excusing circumstances. If the subject actively defends the target's actions as fully correct, it leans towards 'Supportive'.
- "neutral"
  * Core Definition: No clear value judgment or affective stance.
  * Anchor Examples (Non-exhaustive): e.g., objective statements of facts, news-like reports.
  * Boundary Rule: Use ONLY if the expression is purely factual. If there is a deliberate, cynical withdrawal of care, use 'Indifferent'.
- "indifferent"
  * Core Definition: Explicitly conveying a lack of care or engagement regarding the target.
  * Anchor Examples (Non-exhaustive): e.g., "whatever," "doesn't matter," showing apathy.
  * Boundary Rule: The focus is on apathy. If the subject actively downplays the target's importance to shut them down, use 'Dismissive'.
- "disapproving"
  * Core Definition: Negative evaluation of a specific behavior, decision, or choice, while still treating the target as an equal whose overall value is intact.
  * Anchor Examples (Non-exhaustive): e.g., being critical, pointing out flaws, reproachful about a specific issue.
  * Boundary Rule: The focus MUST be on the action/issue, not the person. If the evaluation attacks the target's inherent worth or is delivered from a position of superiority, classify as 'Contemptuous'.
- "skeptical"
  * Core Definition: Holding reservations about the target's authenticity, ability, claims, or motives.
  * Anchor Examples (Non-exhaustive): e.g., doubting assumptions, hinting at unreliability, guarded belief.
  * Boundary Rule: The focus is on doubt. If the subject outright denies the validity or worth of the target without consideration, classify as 'Dismissive'.
- "concerned"
  * Core Definition: Belief that the target might bring risk, harm, or negative consequences.
  * Anchor Examples (Non-exhaustive): e.g., reminding of risks, using a cautious tone, expressing worry.
  * Boundary Rule: The focus is on potential future negative outcomes. If the subject expresses empathy for an already occurred negative outcome, use 'Sympathetic'.
- "dismissive"
  * Core Definition: Denying the importance, reasonableness, or discussion value of the target or their standpoint.
  * Anchor Examples (Non-exhaustive): e.g., brushing off, ignoring, treating the input as trivial, using irony to downplay.
  * Boundary Rule: The core action is ignoring or shutting down. It differs from 'Disapproving' because it doesn't seriously evaluate the action; it differs from 'Contemptuous' because it doesn't necessarily attack the target's dignity.
- "contemptuous"
  * Core Definition: Viewing the target as inherently inferior, worthless, or beneath respect.
  * Anchor Examples (Non-exhaustive): e.g., acting disdainful, mocking, arrogant, patronizing, condescension.
  * Boundary Rule: The focus is on degrading the person. If the subject is merely denying the importance of a topic without personal attacks, map to 'Dismissive'.
- "hostile"
  * Core Definition: Aggressive antagonism, aiming to attack, harm, or dehumanize the target.
  * Anchor Examples (Non-exhaustive): e.g., insults, ugly portrayals, group attacks, explicit threats.
  * Boundary Rule: The focus is on explicit aggression. If the negativity is an expression of superiority without active aggressive attack, use 'Contemptuous'.
```

- Intent 下各部分定义 :

```python
1. "mechanism": 
- "prosocial deception": A benevolent concealment where the speaker prioritizes social harmony or face-saving over factual accuracy. Key signatures: "White lies" (masking true negative feelings like disgust or disappointment with surface positivity to protect the target's feelings or avoid direct rejection).
- "malicious manipulation": A strategic intent to exploit human vulnerabilities while packaging harm or control as help, vulnerability, or morality. Key signatures: "Killing with kindness" (over-praising or indulging to set the target up for a fall), "Playing the victim" (exaggerating one's own suffering to guilt-trip or frame others), or "Moral kidnapping" (using high moral standards to force compliance or stigmatize the target).
- "expressive aggression": Indirect or veiled hostility used to attack, mock, or control the target while maintaining a facade. Key signatures: "Veiled abuse" (publicly mocking or insulting under the guise of giving advice/gifts) or "Implicit threats" (using overly specific observations or fake care to instill psychological pressure and fear).
- "benevolent provocation": A strategic disguise used to trigger a desired action or reveal the truth. Key signatures: "Goading" or "Reverse psychology" (creating a fake scenario, pretending ignorance, or challenging the target to force them to expose their true stance or confess).

2. "label": 
- "mitigate"
  * Core Definition: Intent to de-escalate tension, save face, and maintain a cooperative relationship.
  * Anchor Examples (Non-exhaustive): e.g., offering a compromise, smoothing over an awkward scenario, telling a prosocial "white lie".
  * Boundary Rule: The primary goal MUST be reducing friction or protecting harmony. If the subject uses fake empathy merely to set up an attack later, classify as 'Provoke' or 'Dominate'.
- "intimidate"
  * Core Definition: Intent to coerce the target through fear, implied consequences, or threats.
  * Anchor Examples (Non-exhaustive): e.g., implicit threats, leveraging specific knowledge to instill psychological pressure, forcing compliance.
  * Boundary Rule: The focus is on instilling fear for control. If the aggression is purely for amusement without the goal of coercing behavior, classify as 'Mock'.
- "alienate"
  * Core Definition: Intent to isolate the target by stripping them of equal status and reducing them to an "outsider" or negative stereotype.
  * Anchor Examples (Non-exhaustive): e.g., othering ("us vs. them"), dehumanizing, group-based exclusion.
  * Boundary Rule: The focus is on identity/group exclusion. If the public isolation is based on a specific moral failing rather than identity/stereotyping, map to 'Denounce'.
- "mock"
  * Core Definition: Intent to entertain oneself or others at the target's expense by making them look foolish.
  * Anchor Examples (Non-exhaustive): e.g., treating flaws as a joke, humiliation for amusement, teasing.
  * Boundary Rule: The primary goal is amusement. If the humor/mockery is used as a veil for a severe attack or baiting to trigger anger, classify as 'Provoke'. If the goal is strictly asserting power, map to 'Dominate'.
- "denounce"
  * Core Definition: Intent to socially punish the target by exposing their actions to an audience and rallying social pressure.
  * Anchor Examples (Non-exhaustive): e.g., public shaming, canceling, rallying a crowd against the target.
  * Boundary Rule: MUST involve a public or multi-party audience dimension. If the judgment is purely one-on-one based on rules/ethics without rallying an audience, classify as 'Condemn'.
- "provoke"
  * Core Definition: Intent to bait, anger, or attack the target while maintaining plausible deniability, often using disguised hostility.
  * Anchor Examples (Non-exhaustive): e.g., using meme-based mockery, heavy irony, reverse psychology, goading to force a reaction.
  * Boundary Rule: The focus is on triggering a reaction or attacking from behind a shield of ambiguity. If the hostility is direct and aims to establish a clear hierarchy, map to 'Dominate'.
- "dominate"
  * Core Definition: Intent to establish power or intellectual asymmetry, forcing the target into a subordinate position.
  * Anchor Examples (Non-exhaustive): e.g., pulling rank, acting patronizingly, condescension, "killing with kindness".
  * Boundary Rule: The focus is on establishing a vertical hierarchy ("I am above you"). If the control is achieved primarily through fear rather than status assertion, map to 'Intimidate'.
- "condemn"
  * Core Definition: Intent to judge the target's actions as ethically wrong or invalid from a moral or logical high ground.
  * Anchor Examples (Non-exhaustive): e.g., playing the righteous judge, moral kidnapping, ethical lecturing.
  * Boundary Rule: The focus is strictly on rules and ethics. It differs from 'Denounce' because it doesn't necessarily rely on rallying a public mob, and differs from 'Dominate' because the superiority stems from a moral code rather than pure social rank.
```

> First in first，先确定样本的主体scenario，在确定的scenario中进行后续所有的prompt以及agent工程都在确定的该scenario下运行，即针对不同的scenario采取不同的策略
> 

### 一、 感知层 (Perception Layer)：剥离主观，构建基石

感知层的核心作用是解耦“客观事实”与“主观推断”。LLM在直接处理复杂社交场景时，常常会过早地产生“幻觉”或主观臆断。将感知拆分为显性与上下文，能让模型的认知更有锚点。

#### 模块1：显性感知 (Explicit Perception)

• 如何强化认知： 培养模型的“克制力”。它强制模型在进行任何推理前，先充当一个绝对客观的“记录者”。这种信息过滤机制保证了后续推理的证据链（Evidence Chain）不被早期的主观偏见污染。
• 具体实现思路：
    ◦ 多模态解析提取： 运用多模态大模型对图像、视频或文本进行结构化信息抽取。
    ◦ Prompt工程： 设计严格的Prompt，要求模型仅输出“物理层面可观察的现象”（例如：“说话者眉头紧锁”、“使用了反问句式”），并严禁在这一步输出情绪或动机猜测。
    ◦ 结构化输出： 强制模型以JSON格式输出，字段：

```json
{
	'text_components':{
		'subject':
		'object':
		'predicate':
		'attribute':
		'adverbial':
	}
	'image_action':{
		'subject':
		'background':
		'behavior':
		'action':
	}
	'audio_caption':{
		'subject':
		'object':
		'predicate':
		'attribute':
		'adverbial':
	}
}
```

#### 模块2：社会语境构建 (Social Context Construction)

• 如何强化认知： 赋予模型“社会常识”和“历史记忆”。它让LLM理解，同样一句话在老板和员工之间、或者在不同文化背景下，含义是截然不同的。
• 具体实现思路：
    ◦ 关系图谱构建： 让LLM基于输入信息，构建一个轻量级的局部知识图谱（Subject Relations），明确交互主体之间的权力、亲密度和历史渊源。

### 二、 推理层 (Reasoning Layer)：捕捉冲突，溯因解释

这是CoDAR框架的灵魂所在。它完美契合了心理学中的预期违背理论（EVT），将模糊的“弦外之音”转化为了一个可计算的数学和逻辑问题。

#### 模块3：预期与冲突建模 (Expectation and Conflict Modeling)

• 如何强化认知： 引入了**反事实思维（Counterfactual Thinking）**的雏形。模型需要学会“预测正常情况”，并对“异常情况”保持敏感。这种计算预期与现实之间偏差（$\Delta$）的能力，是理解隐喻、讽刺和高情商表达的关键。
• 具体实现思路：
    ◦ 双路提示 (Dual-Prompting) 机制：
        ▪ *第一路 (Expected)*：基于模块2的Context，询问LLM：“在这个社交场景下，符合常理和规范的表达/行为应该是什么？”（得到变量 $e$）。
        ▪ *第二路 (Conflict)*：将 $e$ 与模块1的真实输出 $x$ 进行对比。
    ◦ 冲突量化： 让模型执行公式 $\mathcal{C} = \Delta(x, e)$。具体来说，就是让LLM输出一份“偏差报告”（Deviation Report），明确指出哪些地方打破了常规预期。

1. 定义冲突类型学
    - 根据每个scenario下的四个mechanism定义conflict
2. 定义表示形式
    - 多维向量 / 多标签槽位 / factor graph / structured JSON，都可以，但必须固定
    - 每个 conflict 至少要有：触发证据、偏离对象、偏离方向、置信度
3. 定义 Δ(x,e)\Delta(x,e)Δ(x,e) 的实现
    - 是 learned scorer
    - 是 rule-augmented comparator
    - 还是 LLM-as-judge 的 structured judgment
    没有这个，方法不可复现，也不可证伪。

#### 模块4：溯因解释 (Abductive Explanation)

• 如何强化认知： 实现了真正的“心智理论（ToM）”。溯因推理（Abductive Reasoning）是指“从现象寻找最佳解释”。通过分析“直接表达真话的成本”，模型能像人类一样理解“社交伪装”的必要性（即印象管理理论）。
• 具体实现思路：
    ◦ 思维链 (Chain-of-Thought, CoT) 设计： 设计一个多步引导的CoT Prompt：
        1. *成本分析*：“如果主体直接表达真实想法，会面临什么社交风险或成本（如破坏关系、损失面子）？”
        2. *动机倒推*：“为了规避上述风险，主体当前的表面行为试图达成什么隐藏目标？”
        3. *机制还原*：“这属于哪种社交策略（如委婉、反讽、转移话题）？”
    ◦ 多假设生成与打分 (Tree of Thoughts, ToT)： 模型可以针对冲突 $\mathcal{C}$ 生成多个解释假设，然后基于上下文为其合理性打分，选取最优解释。

> 
> 
> 
> 在 abductive layer 前面加一个 **gating / null hypothesis stage**：
> 
> 1. 先判断这个样本是否真的存在需要解释的 social conflict
> 2. 再判断“策略性表达”是否优于更简单的解释
> 
> 建议把 hypothesis space 显式扩展为：
> 
> - H0: no meaningful conflict
> - H1: perception error / missing context
> - H2: accidental mismatch / noise
> - H3: strategic social expression
> - H4: culture-specific or in-group code
> - H5: sarcasm / irony / indirect attack / face-saving 等机制性解释
> 
> 然后用 evidence fit + context fit + parsimony 去筛。
> 

### 三、 验证层 (Verification Layer)：逻辑闭环，自我纠错

任何复杂的推理都伴随着极高的出错率，验证层是保证系统鲁棒性（Robustness）的最后一道防线。

#### 模块5：一致性验证 (Consistency Verification)

• 如何强化认知： 赋予模型**元认知（Metacognition）**能力——即“思考自己的思考”。模型不再盲目自信，而是学会了自我审视和逻辑排谬。
• 具体实现思路：
    ◦ Agentic 评测机制： 引入一个独立的“Critic Agent（评论家智能体）”。该Agent不参与生成，只负责检查模块4的结论是否与模块1的“客观事实”和模块2的“社会规则”产生矛盾。—采用辩论框架（Debate Prompting）
    ◦ 回溯机制 (Backtracking)： 一旦Critic Agent发现逻辑不闭环（例如：推断出的隐藏动机无法解释某个微表情），就会生成具体的修改建议，并将错误信息反馈给推理层（模块3或4）进行重新生成（Refinement Loop）。设置最大重试次数以防死循环。

> 引入**多智能体（Multi-Agent）协同或早退机制（Early-exit）**。不要等待所有推理走完才验证。可以让感知Agent和推理Agent并行工作，进行持续的信念更新（Belief Updating）。例如，在生成假设的过程中，一旦发现当前假设与已知证据的置信度低于阈值，立刻在局部截断并重采样，而不是堆积到模块5统一处理。让感知Agent成为“大脑”来调度协调agent pipline，在进入stage4前先判断是否需要Abductive Reasoning
>