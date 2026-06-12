# 09 — 传统MLOps与LLMOps融合

> **定位**: LLMOps体系 · 工程融合深度  
> **更新**: 2026-05-29 · 对标传统ML到LLM的全栈运营  

## 元数据
- **难度**: ⭐⭐⭐
- **前置知识**: MLOps基础, LLM基础
- **关联文件**: [02-可观测性与治理](./02-可观测性与治理.md), [05-LLM SRE与生产运维](./05-LLM%20SRE与生产运维.md)
- **最后更新**: 2026-06-12

---

## 1. MLOps 到 LLMOps — 不是替代，是进化

传统MLOps已发展了6年+，从Google 2015年的Hidden Technical Debt论文到MLflow/Kubeflow成熟生态。LLMOps不是推倒重来——80%的MLOps原则仍然适用，但20%的关键差异决定了架构设计的根本不同。

> 理解传统MLOps，才能真正理解LLMOps为什么"不同"。

**传统ML vs LLM系统的本质差异**:
```
维度            传统ML                LLM系统
─────────────────────────────────────────────────
模型            小/中型(10MB-1GB)     大/超大(1GB-1TB)
训练            需要训练数据          预训练+微调/无需训练
输出            分类/回归/排序        自由文本/代码/结构化
评估            数值指标(F1/AUC)      多维度(人工+自动)
延迟            亚毫秒-毫秒           秒级
硬件            CPU/单GPU            多GPU/GPU集群
版本            模型权重               模型+Prompt+配置
Pipeline        CT (持续训练)          CICD+Prompt版本
成本            固定(硬件)            按Token计费
可解释性        相对容易              极难
```

---

## 2. 传统ML Pipeline 全貌

### 2.1 标准化Pipeline架构

```
传统ML Pipeline (TFX/Kubeflow风格):

┌─────────────────────────────────────────────────────┐
│                    ML Pipeline                       │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ 数据     │→ │ 数据     │→ │ 特征     │→ │ 模型   │ │
│  │ 摄取     │  │ 验证     │  │ 工程     │  │ 训练   │ │
│  │(Ingest) │  │(Validate)│  │(Transform)│ │(Train) │ │
│  └─────────┘  └─────────┘  └─────────┘  └───┬────┘ │
│                                              ↓      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │ 模型     │← │ 模型     │← │ 模型     │← │ 模型   │ │
│  │ 服务     │  │ 推送     │  │ 验证     │  │ 评估   │ │
│  │(Serve)  │  │(Push)   │  │(Validate)│  │(Evaluate)│ │
│  └─────────┘  └─────────┘  └─────────┘  └────────┘ │
│                                                      │
│  ┌─────────────────────────────────────────────────┐│
│  │              实时监控 + 漂移检测                  ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘

关键组件:
1. Feature Store (特征仓库): Feast/Tecton — 特征在线/离线一致性
2. Model Registry (模型注册): MLflow Model Registry — 版本+阶段管理
3. Training Orchestration: Kubeflow/Airflow — 调度+编排
4. Model Serving: TF Serving/TorchServe/Seldon — 推理服务
5. Monitoring: Evidently/WhyLabs — 漂移检测+性能监控
```

### 2.2 特征工程平台 (Feature Store)

```python
# Feast — 最流行的开源Feature Store
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.file import FileOfflineStoreConfig

# 1. 定义实体
driver = Entity(name="driver", join_keys=["driver_id"])

# 2. 定义特征视图 (特征的时间序列版本)
driver_stats = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=30),  # 特征有效期
    schema=[
        Field(name="avg_rating", dtype=Float32),
        Field(name="total_trips", dtype=Int64),
        Field(name="acceptance_rate", dtype=Float32),
    ],
    source=BigQuerySource(
        table="driver_statistics",
        timestamp_field="event_timestamp"
    )
)

# 3. 部署特征
store = FeatureStore(repo_path="./feature_repo")
store.apply([driver, driver_stats])

# 4. 训练时获取历史特征 (时间点正确!)
training_features = store.get_historical_features(
    entity_df=training_events,  # 含driver_id + event_timestamp
    features=[
        "driver_stats:avg_rating",
        "driver_stats:total_trips",
        "driver_stats:acceptance_rate",
    ]
).to_df()

# 5. 推理时获取在线特征 (<10ms)
online_features = store.get_online_features(
    entity_rows=[{"driver_id": 12345}],
    features=[
        "driver_stats:avg_rating",
        "driver_stats:total_trips",
    ]
).to_dict()

# 关键价值: 
# - 训练/推理特征一致性保证 (避免training-serving skew)
# - 时间点正确 (point-in-time correct joins)
# - 特征复用和共享 (避免重复计算)
```

### 2.3 MLflow — 实验管理和模型注册

```python
# MLflow 完整工作流
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 1. 实验跟踪
mlflow.set_experiment("click_through_prediction")

with mlflow.start_run(run_name="xgboost_v3") as run:
    # 记录参数
    mlflow.log_params({
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
    })
    
    # 训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 记录指标
    y_pred = model.predict(X_test)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    })
    
    # 记录模型
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="click_through_model"
    )
    
    # 记录特征重要性
    fig = plot_importance(model)
    mlflow.log_figure(fig, "feature_importance.png")

# 2. 模型注册和阶段管理
client = MlflowClient()

# 获取最新版本
latest_versions = client.get_latest_versions("click_through_model")
latest_version = latest_versions[0].version

# 逐步推进: None → Staging → Production → Archived
client.transition_model_version_stage(
    name="click_through_model",
    version=latest_version,
    stage="Staging"
)

# 生产部署前验证
staging_model = mlflow.pyfunc.load_model(
    f"models:/click_through_model/Staging"
)

# 验证通过 → 推到Production
client.transition_model_version_stage(
    name="click_through_model",
    version=latest_version,
    stage="Production",
    archive_existing_versions=True  # 归档旧版本
)

# 3. 模型加载 (生产环境)
production_model = mlflow.pyfunc.load_model(
    f"models:/click_through_model/Production"
)
```

---

## 3. 模型监控与漂移检测

### 3.1 传统ML的漂移检测

```python
# Evidently AI — 数据/模型漂移检测
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
)

# 1. 数据漂移检测
data_drift_report = Report(metrics=[
    DataDriftTable(),
    DatasetDriftMetric(),
    ColumnDriftMetric(column_name="price"),
    ColumnDriftMetric(column_name="age"),
])

data_drift_report.run(
    reference_data=training_data,  # 训练时的特征分布
    current_data=production_data,  # 当前的特征分布
)

# 关键漂移指标:
# - PSI (Population Stability Index): 分布偏移量，<0.1正常, 0.1-0.25警告, >0.25严重
# - KS (Kolmogorov-Smirnov): 两分布最大差异
# - Jensen-Shannon Divergence: 更平滑的距离度量
# - Wasserstein Distance: 考虑分布形状

# 2. 预测漂移检测
prediction_drift = Report(metrics=[
    ColumnDriftMetric(column_name="prediction"),
    ColumnDriftMetric(column_name="prediction_probability"),
])

prediction_drift.run(
    reference_data=training_predictions,
    current_data=current_predictions,
)

# 3. 性能漂移 (需要标签)
performance_report = Report(metrics=[
    RegressionPreset(),  # 回归模型
    # ClassificationPreset(),  # 分类模型
])

performance_report.run(
    reference_data=eval_data,
    current_data=recent_data_with_labels,
)

# 告警规则
if data_drift_report.dataset_drift():
    send_alert("特征分布发生显著漂移，建议触发重训练")
```

### 3.2 传统ML vs LLM 监控对比

| 监控维度 | 传统ML | LLM系统 |
|----------|--------|---------|
| **数据漂移** | 特征分布变化 (KS/PSI) | Prompt分布变化、用户意图变化 |
| **模型漂移** | 预测分布变化 | Token分布变化、输出风格变化 |
| **性能退化** | 准确率/F1/AUC下降 | 人工评估分下降、用户满意度↓ |
| **业务指标** | 转化率/收入影响 | Token消耗/GPT-4使用率/成本 |
| **延迟漂移** | p95延迟变化 | TTFT/TPOT变化 |
| **数据质量** | 缺失值/异常值/范围检查 | 截断/编码问题/注入攻击 |
| **公平性** | 群体间准确率差异 | 群体间输出质量和拒绝率差异 |

---

## 4. LLMOps 扩展传统MLOps

### 4.1 Prompt版本管理

```python
# Prompt Registry (类比Model Registry)
class PromptRegistry:
    """
    Prompt的版本管理和A/B测试
    类比: 传统ML的Model Registry
    """
    
    def __init__(self):
        self.prompts: Dict[str, List[PromptVersion]] = {}
    
    def register_prompt(self, name: str, template: str, 
                        variables: List[str], metadata: dict) -> str:
        """注册新Prompt版本"""
        version = f"v{len(self.prompts.get(name, [])) + 1}"
        
        prompt_version = PromptVersion(
            name=name,
            version=version,
            template=template,
            variables=variables,
            metadata={
                **metadata,
                "created_at": datetime.now(),
                "hash": hashlib.sha256(template.encode()).hexdigest()
            }
        )
        
        if name not in self.prompts:
            self.prompts[name] = []
        self.prompts[name].append(prompt_version)
        
        return version
    
    def get_prompt(self, name: str, version: str = "latest") -> PromptVersion:
        """获取指定版本的Prompt"""
        if version == "latest":
            return self.prompts[name][-1]
        
        for pv in self.prompts[name]:
            if pv.version == version:
                return pv
        
        raise ValueError(f"Prompt {name}:{version} not found")
    
    def ab_test(self, name: str, version_a: str, version_b: str, 
                split_ratio: float = 0.5):
        """Prompt A/B测试"""
        return PromptABTest(
            prompt_name=name,
            variant_a=self.get_prompt(name, version_a),
            variant_b=self.get_prompt(name, version_b),
            split_ratio=split_ratio,
            metrics=["user_satisfaction", "task_completion", "latency"]
        )

# 使用示例
registry = PromptRegistry()

# 注册多个版本
registry.register_prompt(
    name="customer_support",
    template="你是{company}的客服助手。用户问题: {query}。请礼貌、专业地回答。",
    variables=["company", "query"],
    metadata={"author": "alice", "description": "基础客服prompt"}
)

registry.register_prompt(
    name="customer_support",
    template="""你是{company}的高级客服专家。
用户背景: {user_context}
用户问题: {query}
请遵循以下原则:
1. 先共情，再解决问题
2. 提供具体可操作的步骤
3. 必要时询问更多信息""",
    variables=["company", "user_context", "query"],
    metadata={"author": "bob", "description": "增强版客服prompt，增加共情和个性化"}
)

# A/B测试
registry.ab_test("customer_support", "v1", "v2")
```

### 4.2 LLM评估体系 — 超越数值指标

```python
# LLM评估多维体系
class LLMEvaluationPipeline:
    """
    传统ML: 单一数值指标 (F1/AUC)
    LLM: 多维评估 (自动+人工+对比)
    """
    
    def run_full_evaluation(self, model, test_cases: List[TestCase]) -> EvalReport:
        results = {}
        
        # 1. 自动指标
        results["automatic"] = {
            "rouge_l": rouge.compute(predictions=outputs, references=references),
            "bert_score": bert_score.compute(predictions=outputs, references=references),
            "semantic_similarity": self._compute_semantic_sim(outputs, references),
            "faithfulness": self._evaluate_faithfulness(outputs, contexts),
            "hallucination_rate": self._detect_hallucinations(outputs, contexts),
        }
        
        # 2. LLM-as-Judge
        results["llm_judge"] = {
            "helpfulness": self._llm_judge_scoring(outputs, "helpfulness", 1-5),
            "accuracy": self._llm_judge_scoring(outputs, "accuracy", 1-5),
            "safety": self._llm_judge_scoring(outputs, "safety", 1-5),
            "coherence": self._llm_judge_scoring(outputs, "coherence", 1-5),
        }
        
        # 3. 对比评估
        results["comparative"] = self._compare_with_baseline(
            current_model=model,
            baseline_model=self.baseline_model,
            test_cases=test_cases[:50]  # 抽样子集
        )
        
        # 4. 安全评估
        results["safety"] = self._run_safety_eval(model)
        
        # 5. 成本效率
        results["efficiency"] = {
            "avg_tokens_per_response": self._compute_avg_tokens(outputs),
            "avg_latency": self._compute_avg_latency(timings),
            "cost_per_1k_requests": self._compute_cost(model, test_cases),
        }
        
        return EvalReport(results)
```

---

## 5. 混合系统架构 — 传统ML + LLM

### 5.1 推荐系统的混合架构

```python
# 推荐系统: 传统ML召回 + LLM精排/解释
class HybridRecommendationSystem:
    """
    混合推荐架构:
    Layer 1 (传统ML): 召回 + 粗排 — 处理百万级候选
    Layer 2 (传统ML): 精排 — 上千候选打分
    Layer 3 (LLM): 个性化解释和对话式推荐
    """
    
    def __init__(self):
        # 传统ML组件
        self.recall_model = TwoTowerModel()        # 双塔召回
        self.rank_model = XGBoostRanker()          # XGBoost精排
        self.feature_store = FeastFeatureStore()    # 特征仓库
        
        # LLM组件
        self.llm = ChatOpenAI(model="gpt-4o-mini")
    
    async def recommend(self, user_id: str, context: dict) -> Recommendation:
        # Stage 1: 召回 (传统ML, <50ms)
        user_embedding = await self.feature_store.get_user_embedding(user_id)
        candidates = await self.recall_model.retrieve(
            user_embedding, 
            top_k=1000
        )
        
        # Stage 2: 精排 (传统ML, <20ms)
        features = await self.feature_store.get_features(
            user_id, 
            [c.id for c in candidates]
        )
        scores = self.rank_model.predict(features)
        top_items = self._get_top_k(candidates, scores, k=10)
        
        # Stage 3: 个性化解释 (LLM, ~500ms)
        if context.get("need_explanation"):
            explanations = await self._generate_explanations(
                user_id, top_items[:3], context
            )
        else:
            explanations = None
        
        return Recommendation(
            items=top_items,
            scores=scores[:10],
            explanations=explanations,
            latency_breakdown={
                "recall_ms": recall_time,
                "rank_ms": rank_time,
                "explanation_ms": llm_time if explanations else 0
            }
        )
    
    async def _generate_explanations(self, user_id, items, context):
        """用LLM生成个性化推荐理由"""
        user_profile = await self._get_user_summary(user_id)
        
        prompt = f"""基于以下用户画像和推荐物品，为每个推荐生成自然的解释：

用户画像: {user_profile}

推荐物品:
{self._format_items(items)}

要求:
1. 每个解释1-2句话，自然口语化
2. 结合用户的具体偏好
3. 不要直接说"因为算法推荐""""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
```

### 5.2 风控系统的混合架构

```python
# 风控系统: 传统规则+ML+LLM 三层防御
class HybridRiskControl:
    """
    三层防御体系:
    Layer 1: 规则引擎 (毫秒级, 拦截明显的欺诈)
    Layer 2: 传统ML模型 (毫秒级, 评分+分类)
    Layer 3: LLM (秒级, 复杂案例的深度分析和决策)
    """
    
    def evaluate_transaction(self, transaction: Transaction) -> RiskAssessment:
        risk_scores = {}
        
        # Layer 1: 规则引擎 (<5ms)
        rule_result = self.rule_engine.evaluate(transaction)
        if rule_result.is_blocked:
            return RiskAssessment(
                decision="BLOCK",
                reason=rule_result.reason,
                layer="rule_engine"
            )
        risk_scores["rule"] = rule_result.risk_score
        
        # Layer 2: 传统ML模型 (<20ms)
        features = self.feature_extractor.extract(transaction)
        ml_score = self.risk_model.predict_proba(features)[0][1]
        risk_scores["ml"] = ml_score
        
        # 低风险交易直接通过
        if ml_score < 0.3 and risk_scores["rule"] < 0.2:
            return RiskAssessment(
                decision="APPROVE",
                risk_score=ml_score,
                layer="ml_model"
            )
        
        # Layer 3: LLM深度分析 (仅对中高风险交易, ~800ms)
        if ml_score > 0.3 or rule_result.flags:
            llm_result = self._deep_analysis(transaction, features, risk_scores)
            
            final_score = 0.5 * ml_score + 0.5 * llm_result.adjusted_score
            final_decision = "BLOCK" if final_score > 0.7 else "REVIEW" if final_score > 0.4 else "APPROVE"
            
            return RiskAssessment(
                decision=final_decision,
                risk_score=final_score,
                layer="hybrid",
                explanation=llm_result.explanation,
                ml_features=features,
            )
        
        return RiskAssessment(decision="APPROVE", risk_score=ml_score, layer="ml_model")
    
    def _deep_analysis(self, transaction, features, scores) -> LLMRiskResult:
        """LLM进行复杂案例分析"""
        prompt = f"""分析以下交易的风险：

交易信息:
- 金额: {transaction.amount}
- 时间: {transaction.timestamp}
- 商户: {transaction.merchant}
- 地点: {transaction.location}
- 设备: {transaction.device_fingerprint}

ML模型风险评分: {scores['ml']:.2f}
规则引擎标记: {scores['rule']:.2f}

请分析:
1. 是否存在欺诈模式？
2. 调整风险评分的理由
3. 建议的处置方式: 通过/人工审核/拒绝

输出JSON格式:
{{
    "adjusted_score": 0.0-1.0,
    "risk_factors": ["因素1", "因素2"],
    "recommendation": "APPROVE|REVIEW|BLOCK",
    "explanation": "..."
}}"""
        
        response = self.llm.invoke(prompt)
        return self._parse_llm_response(response.content)
```

---

## 6. A/B实验平台

### 6.1 传统 vs LLM A/B测试差异

```
传统ML A/B测试:              LLM A/B测试:

变量:                        变量:
├── 模型版本                  ├── 模型 (GPT-4o vs Claude)
├── 特征工程                  ├── Prompt模板
├── 超参数                    ├── 温度/top_p等参数
└── 模型架构                  ├── RAG策略
                              ├── Few-shot示例
                              └── System Prompt

指标:                        指标:
├── 业务指标 (转化率/GMV)     ├── 业务指标
├── 模型指标 (准确率)         ├── LLM质量指标 (人工评估)
                              ├── Token消耗/成本
                              └── 用户体验 (满意度/重试率)
```

### 6.2 LLM A/B测试框架

```python
# LLM A/B测试平台
class LLMABTestPlatform:
    """
    支持多层实验:
    - Prompt A/B测试
    - 模型版本A/B测试
    - RAG策略A/B测试
    """
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.assignment_cache = {}
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """创建A/B实验"""
        exp_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        experiment = Experiment(
            id=exp_id,
            name=config.name,
            variants=config.variants,
            traffic_split=config.traffic_split,  # {"control": 0.5, "treatment": 0.5}
            metrics=config.metrics,
            duration_days=config.duration_days,
            min_sample_size=config.min_sample_size,
        )
        
        self.experiments[exp_id] = experiment
        return exp_id
    
    def assign_variant(self, user_id: str, experiment_id: str) -> str:
        """用户分桶 (一致性哈希，确保同一用户始终在同一组)"""
        
        # 使用一致性哈希确保用户体验稳定
        hash_key = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16) % 100
        
        experiment = self.experiments[experiment_id]
        
        cumulative = 0
        for variant, percentage in experiment.traffic_split.items():
            cumulative += percentage * 100
            if hash_value < cumulative:
                return variant
        
        return list(experiment.traffic_split.keys())[-1]  # fallback
    
    def log_metric(self, experiment_id: str, user_id: str, 
                   variant: str, metrics: dict):
        """记录实验指标"""
        event = ExperimentEvent(
            experiment_id=experiment_id,
            user_id=user_id,
            variant=variant,
            timestamp=datetime.now(),
            metrics=metrics,
        )
        self.event_store.append(event)
    
    def analyze_results(self, experiment_id: str) -> ExperimentReport:
        """统计分析实验效果"""
        experiment = self.experiments[experiment_id]
        events = self.event_store.get_by_experiment(experiment_id)
        
        # 分组统计
        results = {}
        for variant in experiment.variants:
            variant_events = [e for e in events if e.variant == variant]
            results[variant] = {
                "sample_size": len(variant_events),
                "metrics": self._aggregate_metrics(variant_events, experiment.metrics)
            }
        
        # 统计显著性检验
        significance = {}
        for metric in experiment.metrics:
            control_values = [e.metrics[metric] for e in events if e.variant == "control"]
            treatment_values = [e.metrics[metric] for e in events if e.variant == "treatment"]
            
            # t-test for continuous metrics
            t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
            significance[metric] = {
                "p_value": p_value,
                "is_significant": p_value < 0.05,
                "effect_size": (np.mean(treatment_values) - np.mean(control_values)) / np.std(control_values),
            }
        
        return ExperimentReport(
            experiment_id=experiment_id,
            results=results,
            significance=significance,
            recommendation=self._generate_recommendation(results, significance),
        )
```

---

## 7. 从传统MLOps迁移到LLMOps的路线图

```
迁移五阶段:

阶段1: 评估和规划 (2-3周)
├── 盘点现有MLOps基础设施
├── 识别LLM使用场景
├── 制定混合治理策略
└── 选择首个试点项目 (低风险、高价值)

阶段2: 基础LLMOps (4-6周)
├── Prompt管理 (版本控制+注册中心)
├── LLM评估框架 (自动+人工)
├── 基础监控 (Token使用+延迟+成本)
└── 安全护栏 (内容过滤+注入检测)

阶段3: 混合管道 (4-6周)
├── 改造Feature Store支持LLM上下文
├── 扩展Model Registry支持Prompt
├── 整合传统ML+LLM的A/B测试
└── 统一监控面板

阶段4: 高级能力 (6-8周)
├── 语义缓存层
├── 模型路由和故障转移
├── FinOps成本管理
└── 自动化回归测试

阶段5: 优化和规模化 (持续)
├── 持续优化Prompt和模型选择
├── 建立LLM团队最佳实践
├── 合规审计自动化
└── 知识库和文化建设
```

---

## 8. 统一运维工具链全景

```
传统ML + LLM 统一工具链:

┌──────────────────────────────────────────────────────────┐
│                     统一管理层                            │
│  MLflow / Weights & Biases / Neptune                      │
│  → 实验跟踪 + 模型注册 + Prompt注册                      │
├──────────────────────────────────────────────────────────┤
│                     管道编排层                            │
│  Kubeflow Pipelines / Airflow / Prefect                   │
│  → ML Pipeline + LLM Eval Pipeline + 部署Pipeline        │
├───────────────────────┬──────────────────────────────────┤
│     传统ML服务层       │        LLM服务层                 │
│  TF Serving           │  vLLM / LiteLLM Proxy            │
│  TorchServe           │  LangServe                       │
│  Seldon Core          │  Custom FastAPI                  │
│  BentoML              │                                  │
├───────────────────────┴──────────────────────────────────┤
│                     特征/数据层                           │
│  Feast / Tecton       │  Pinecone / Milvus / Weaviate   │
│  (特征仓库)            │  (向量数据库)                    │
├──────────────────────────────────────────────────────────┤
│                     监控层                                │
│  Evidently AI / WhyLabs / LangKit / Phoenix              │
│  → 数据漂移 + 模型漂移 + LLM质量 + Token成本             │
├──────────────────────────────────────────────────────────┤
│                     基础设施层                            │
│  Kubernetes / Ray / GPU Operator                         │
│  → 统一调度 CPU任务(ML训练) + GPU任务(LLM推理)           │
└──────────────────────────────────────────────────────────┘
```

## 深度分析

### 传统MLOps与LLMOps的融合不是叠加而是重构

简单地将传统MLOps工具加上LLM功能（比如在MLflow里加Prompt版本管理）是不够的。真正的融合需要在架构层面重新思考：Feature Store如何扩展支持LLM上下文向量？Model Registry如何同时管理模型权重和Prompt配置？实验跟踪如何记录Prompt变体、温度参数和RAG策略？核心原则是"统一元数据层 + 差异化执行引擎"——用同一个元数据平台管理ML模型和LLM应用，但针对LLM的特性（概率性输出、Token消耗、Prompt敏感等）设计独立的评估和监控机制。

### 混合系统的架构模式

2025-2026年最成功的AI应用架构是"传统ML + LLM"的混合模式。典型模式包括：(1) 传统ML做粗筛（毫秒级召回/分类），LLM做精加工（秒级生成/推理）；(2) 传统ML做高频预测（如推荐排序），LLM做低频的个性化解释和交互；(3) 传统ML做风控规则执行，LLM做复杂案例的深度分析和决策。关键原则是"让传统ML做它擅长的事（速度、确定性、可解释），让LLM做它擅长的事（语言理解、生成、推理）"。

### 迁移路线图的关键抉择

从传统MLOps迁移到融合LLMOps需要做三个关键抉择：(1) 工具链——是扩展现有MLflow/Kubeflow还是引入新工具（Langfuse/Phoenix）？建议采取混合策略：保留MLflow做模型管理，增加Langfuse做LLM评估和追踪；(2) 团队——是现有ML团队学习LLM还是组建独立LLM团队？建议在现有ML团队中设立"LLM大使"角色，避免知识孤岛；(3) 治理——是否统一ML和LLM的审批流程？建议统一模型上线审批，但LLM额外增加Prompt审查和安全评估环节。

## Checklist

- [ ] 盘点现有MLOps基础设施（Feature Store / Model Registry / Pipeline）
- [ ] 扩展MLflow支持LLM元数据（Prompt版本、评估分数、Token消耗）
- [ ] 实现Prompt Registry（类比Model Registry的版本+阶段管理）
- [ ] 建立LLM多维评估体系（自动+人工+LLM-as-Judge）
- [ ] 改造Feature Store支持RAG上下文管理
- [ ] 部署混合A/B测试框架（同时支持ML模型和LLM Prompt实验）
- [ ] 整合传统ML + LLM统一监控面板
- [ ] 实现语义缓存层（降低LLM调用成本）
- [ ] 制定MLOps→LLMOps迁移路线图和优先级
- [ ] 培训团队LLM工程实践（Prompt设计/评估/安全）

## 延伸阅读

- [02-可观测性与治理](./02-可观测性与治理.md) — LLM监控与评估基础设施
- [05-LLM SRE与生产运维](./05-LLM%20SRE与生产运维.md) — 生产运维与GPU容量规划
- MLflow: https://mlflow.org — 实验跟踪和模型注册
- Feast: https://feast.dev — 开源Feature Store
- "Hidden Technical Debt in Machine Learning Systems" — Google NIPS 2015

---

## 参考资源

- MLflow: https://mlflow.org
- Feast: https://feast.dev
- Kubeflow: https://www.kubeflow.org
- Evidently AI: https://www.evidentlyai.com
- TFX: https://www.tensorflow.org/tfx
- BentoML: https://www.bentoml.com
- Google MLOps Guide: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

---

*上一篇: [AI合规与法律框架](./08-AI合规与法律框架.md)*

*最后更新：2026-06-12*
