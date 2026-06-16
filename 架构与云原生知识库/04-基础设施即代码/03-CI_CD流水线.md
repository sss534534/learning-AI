# CI/CD 流水线

> 持续集成和持续交付是现代软件工程的核心实践，自动化构建、测试和部署流程，缩短交付周期。

---

## 目录

1. [CI/CD基础](#1-cicd基础)
2. [CI/CD工具对比](#2-cicd工具对比)
3. [流水线设计](#3-流水线设计)
4. [高级实践](#4-高级实践)

---

## 1. CI/CD基础

### 1.1 核心流程

```
代码提交 → 自动构建 → 自动测试 → 自动部署 → 运行监控
    ↑                                          |
    └──────────────── 回滚 ──────────────────┘
```

### 1.2 不同层级

| 层级 | 范围 | 频次 | 质量门禁 |
|------|------|------|----------|
| CI | 每次提交 | 高频 | 单元测试，lint |
| CD | 合并到主分支 | 中频 | 集成测试，安全扫描 |
| 生产部署 | 发布版本 | 低频 | 全量测试，审批 |

---

## 2. CI/CD工具对比

| 工具 | 托管 | 配置方式 | 特点 |
|------|------|----------|------|
| GitHub Actions | 云 | YAML | 生态丰富，Marketplace |
| GitLab CI | 云/自管 | YAML | 内置Registry |
| Jenkins | 自管 | Groovy/UI | 灵活，插件丰富 |
| ArgoCD | K8s原生 | YAML | GitOps |
| Drone | 自管 | YAML | 轻量，容器化 |

---

## 3. 流水线设计

### 3.1 多阶段流水线

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - test
  - build
  - deploy

lint:
  stage: lint
  script: make lint

test:
  stage: test
  script: make test
  coverage: '/TOTAL.*\s(\d+\.\d+\%)/'

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy_staging:
  stage: deploy
  script: kubectl set image deployment/app app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  environment: staging
  only:
    - main

deploy_production:
  stage: deploy
  script: ./deploy_prod.sh
  environment: production
  when: manual  # 手动触发
  only:
    - tags
```

### 3.2 质量门禁

| 门禁类型 | 工具 | 拦截条件 |
|---------|------|----------|
| 代码风格 | ruff, ESLint | 不合规则失败 |
| 单元测试 | pytest, jest | 覆盖率<80% |
| 安全扫描 | Snyk, Trivy | 高危漏洞 |
| 构建验证 | docker build | 镜像构建失败 |
| 集成测试 | Playwright | E2E失败 |

---

## 4. 高级实践

### 4.1 发布策略

| 策略 | 描述 | 风险 |
|------|------|------|
| 滚动更新 | 逐个替换实例 | 兼容性问题 |
| 蓝绿部署 | 两套环境切换 | 资源翻倍 |
| 金丝雀发布 | 小比例流量验证 | 需要流量管理 |
| 功能开关 | 代码不发布，功能开关控制 | 技术债务 |

### 4.2 部署自动化可靠性

```python
# 渐进式自动化部署脚本
def canary_deploy(new_version, canary_percent=10):
    # 1. 部署金丝雀
    deploy(new_version, replicas=1)

    # 2. 路由10%流量
    route_traffic(percent=canary_percent)

    # 3. 监控指标
    if check_metrics(duration="5m"):
        # 4. 逐步放量
        for pct in [25, 50, 75, 100]:
            route_traffic(percent=pct)
            wait_and_check(duration="2m")
        # 5. 全量完成
        remove_old_version()
    else:
        # 回滚
        rollback()
```

*最后更新：2026-06-15*
