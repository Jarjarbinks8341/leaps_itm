---
name: buildAaplLeaps
description: AAPL LEAPS Backtest — Strategy Reference. Dual-mode entry signals (bottom-divergence reversal + trend-pullback continuation), BSM option pricing, tiered exits, and the auto-research protocol for tuning parameters against historical data.
---

# AAPL LEAPS Backtest — Strategy Reference

> Living document for the `leaps_itm` codebase (`~/repo/leaps_itm/`)。
> 目标：为 Apple（后续扩展 Microsoft、Google）建立自动化 LEAPS 买入信号系统，并通过 auto-research 用历史数据验证。
> 本文档先于代码存在——先定义策略规则，再按规则实现回测。

---

## 核心投资哲学

### 为什么是 LEAPS？

LEAPS（Long-term Equity Anticipation Securities）是到期日超过一年的深度实值看涨期权。相比直接持有正股：

- **内置杠杆**：delta ≈ 0.6，用约 60% 的资金控制 100% 的价格敞口
- **时间容错**：1 年+ 的到期日给市场足够时间回归，不会因短期波动归零
- **风险上限**：最大损失为权利金，而非本金的 100%

### 单一股票 vs 指数：AAPL 与 QQQ 的关键差异

QQQ 版策略（见 `buildLeaps` skill）不能直接照搬到个股，三个结构性差异必须处理：

1. **财报事件风险**：AAPL 每季度财报前 IV 拉升、财报后 IV crush。指数没有这个问题。
   → 引入**财报静默期**（earnings blackout）：财报前 N 天禁止开新仓。
2. **个股特异性风险**：单一公司可能出现与大盘无关的暴跌（诉讼、产品失败、管理层变动）。
   → VIX 只反映大盘恐慌，个股信号需要同时参考大盘状态与个股自身状态。
3. **IV 结构不同**：个股 IV 常年高于指数 IV，且对财报周期敏感。
   → 回测中用 AAPL 自身 30 日实现波动率作为 IV 代理（而非 VIX 衍生值）。

### 双模式入场：反转与回调是两个不同的问题

经过讨论（含 Gemini 建议的多指标共振模型），确认两类买点回答的是**不同的问题**，不能合并为一个 AND 滤网：

- **Mode A（底背离反转）**：「下跌趋势是否已经衰竭？」——在恐慌中买入，期权贵但价格低
- **Mode B（趋势回调续涨）**：「上升趋势中的回调是否可以上车？」——在平静中买入，期权便宜但价格不是最低

两者的市场状态互斥（A 要求恐慌/高 IV，B 天然发生在低恐慌/低 IV 环境），强行 AND 会导致历史上几乎无信号。**正确做法：分别回测，再测试 OR 组合。**

---

## 策略规则（完整版）

### Mode A — 底背离反转（移植自 QQQ 版，已验证的设计）

入场条件（AND，全部满足）：

1. **MACD 底背离（日线，AAPL 价格）**
   - 将过去 `div_lookback` 天分为两个窗口（以 `div_min_gap` 为分界）
   - 前段最低价 vs 后段最低价：**后段价格更低**（lower low）
   - 对应时刻的 MACD histogram：**后段 histogram 更高**（higher low）
   - 两个 histogram 值均为负数（确认处于空头区域）
2. **VIX 位置过滤**
   - `VIX_today > VIX.rolling(vix_ma).mean()_today`
   - 大盘恐慌确认——个股底背离 + 市场恐惧 = 高概率错杀区域
3. **财报静默期**
   - 距下一次 AAPL 财报日 > `earnings_blackout` 天（默认 7 天）

### Mode B — 趋势回调续涨（新增，源自 MA/MACD 共振讨论）

入场条件（AND，全部满足）：

1. **趋势确认**：`MA20 > MA50`（中期趋势向上，只在上升趋势中做回调买入）
2. **回踩企稳**：过去 `pullback_lookback` 天内价格曾触及/跌破 MA20（`low ≤ MA20 × (1 + touch_tolerance)`），且当日收盘价 ≥ MA20
3. **短期动能拐头**：MA5 今日值 > MA5 昨日值（拐头向上），且当日收盘 > MA5
4. **MACD 动能收敛**：MACD histogram 为负但连续 `hist_converge_days` 天抬高（收敛待金叉），或已在零轴上方金叉不超过 3 天
5. **波动率过滤（B 模式专属，与 A 相反）**：30 日实现波动率处于过去一年分位 < `iv_rank_max`（默认 40%）——期权相对便宜时才用 B 模式买入
6. **财报静默期**：同 Mode A

### 模式组合

回测分三种配置分别跑分：

| 配置 | 信号 | 用途 |
|------|------|------|
| `mode=A` | 仅底背离 | 基线（与 QQQ 版可比） |
| `mode=B` | 仅回调续涨 | 验证新模式独立价值 |
| `mode=AB` | A OR B（任一触发即买入，交易记录标注来源） | 检验组合是否优于单一模式 |

`mode=AB` 时按持仓来源分别统计胜率/收益，判断哪个模式贡献了 alpha。

### 仓位管理

**实现已从早期草案的固定 `max_pos`/`pos_pct` 模型，改为移植自 QQQ 版当前代码的动态部署模型（更成熟、已验证）：**

| 规则 | 说明 |
|------|------|
| 每笔仓位 | `lot_pct × 当日 NAV`（默认 5%），Mode A 按 `signal_strength()` 在 `lot_pct`–`lot_pct_max` 间动态放大；Mode B 固定用 `lot_pct` |
| 部署上限 | 期权总市值 ≤ `max_deploy_pct × NAV`（默认 80%），超出时 FIFO 平掉最早仓位腾出空间 |
| DTE 主动平仓 | 剩余到期时间 < `min_months_remaining`（默认 6 个月）时无条件平仓，避免持有临近到期的合约 |
| 最短持有期 | `min_hold_months`（默认 3 个月）内不触发止盈，给信号兑现留出时间 |
| 期权参数 | Delta ≈ `target_delta`（默认 0.6），DTE = `dte_days`（默认 365 天） |
| 同日多信号 | A、B 同日触发只开一仓，记为 A（更稀有、更高确信度） |
| 部分止盈 | `tier1`/`tier2` 支持按 `tpN_close_pct` 部分平仓（默认 100%，可调为分批止盈） |

### 期权选择（每次入场）

1. 用 AAPL 30 日实现波动率计算当前 sigma（floor 5%，防 BSM 数值异常）
2. 反向 BSM：二分搜索找到使 delta = `target_delta` 的行权价 K
3. 用 BSM 计算该合约的当日理论价格（入场权利金）
4. 以 `pos_pct × NAV` 的资金买入

### 出场规则（阶梯止盈 + 强制平仓）

每个交易日对所有持仓检查（先出场，后入场，同日可平仓+开仓）：

```
if months_held > force_months:                              → 强制平仓（无论盈亏）
elif months_held ≤ tier1_months and pnl ≥ tier1_profit:     → tier1 止盈
elif months_held ≤ tier2_months and pnl ≥ tier2_profit:     → tier2 止盈
elif months_held ≤ tier3_months and pnl ≥ tier3_profit:     → tier3 止盈
```

默认值（可调参）：

| 阶段 | 持有时长 | 目标收益 |
|------|---------|---------|
| tier1（蜜月期） | ≤ 4 个月 | +50% |
| tier2 | ≤ 6 个月 | +30% |
| tier3 | ≤ 9 个月 | +10% |
| force（强制） | > 9 个月 | 无论盈亏 |

**候选增强（auto-research 阶段测试，不进 v1 基线）：** MA5/MA20 死叉作为提前出场触发器——趋势明确反转时不等阶梯止盈直接平仓。仅当基线回测显示出场普遍偏晚时才引入。

---

## 期权定价代理：BSM + 30 日实现波动率

历史期权价格数据不可得，用 Black-Scholes-Merton 模型 + AAPL 过去 30 日实现波动率作为 IV 代理，动态计算每日理论期权价格。

**误差来源（已知，可接受）：**
- 真实 IV 通常高于实现波动率约 5–10%（波动率风险溢价），个股比指数更明显
- 忽略买卖价差、IV 微笑/偏斜
- 财报前 IV 拉升无法用实现波动率捕捉 → 财报静默期部分缓解此误差

---

## 数据需求

| 数据 | 来源 | 用途 |
|------|------|------|
| AAPL 日线 OHLCV | yfinance → parquet 缓存 | 价格信号、实现波动率 |
| VIX 日线收盘 | yfinance（`^VIX`） | Mode A 恐慌过滤 |
| AAPL 历史财报日 | yfinance `Ticker.get_earnings_dates(limit=80)`（实测可回溯至 2002 年）→ CSV 缓存 | 财报静默期 |

回测区间：训练集 2015-01-01 ~ 2024-12-31（含 2018 修正、2020 疫情崩盘、2022 加息熊市），测试集 2025-01-01 ~ 今（完全 OOS）。

---

## 架构（已实现）

```
leaps_itm/                     ← git root (~/repo/leaps_itm/)
├── strategy/
│   ├── data.py                — fetch & cache AAPL + VIX + earnings dates
│   ├── signals.py             — compute_macd(), bullish_divergence(), vix_elevated(), signal_strength(),
│   │                            pullback_entry(), iv_rank(), in_earnings_blackout()
│   ├── options.py             — call_price(), call_delta(), strike_for_delta(), realized_vol()
│   ├── portfolio.py           — Position/Trade（含 signal_mode 标注）, Portfolio.step(), FIFO
│   └── metrics.py             — cagr(), max_drawdown(), sharpe(), calmar(), win_rate(), score(), summary_by_mode()
├── backtest.py                — 回测入口（run() + CLI，--mode A|B|AB）
├── optimize.py                — 随机参数搜索入口（--mode A|B|AB，--refine）
├── .claude/skills/buildAaplLeaps/SKILL.md  — 本文档
└── pyproject.toml             — uv 项目依赖
```

`options.py`、`metrics.py`、`portfolio.py` 直接移植自 `~/repo/leaps/leaps/`（QQQ 版最新代码，非早期文档描述的版本——已含动态 lot 定价、DTE 主动平仓、部分止盈等后续演进）。`signals.py` 和 `data.py` 按 Mode B + 财报静默期扩展。

---

## 参数搜索空间

```python
PARAM_GRID = {
    # 共用
    "target_delta":         [0.50, 0.55, 0.60, 0.65, 0.70],
    "dte_days":             [300, 330, 365, 400, 430],
    "lot_pct":              [0.03, 0.05, 0.07, 0.10],
    "lot_pct_max":          [0.10, 0.15, 0.20, 0.25],   # Mode A dynamic-size ceiling
    "min_months_remaining": [3, 4, 5, 6],               # DTE proactive-exit threshold
    "min_hold_months":      [1, 2, 3, 4],
    "earnings_blackout":    [0, 5, 7, 10],          # 0 = 关闭静默期（验证其必要性）
    "tier1_months":         [3, 4, 5],
    "tier1_profit":         [0.30, 0.40, 0.50, 0.60],
    "tier2_months":         [5, 6, 7],
    "tier2_profit":         [0.20, 0.25, 0.30],
    "tier3_months":         [8, 9],
    "tier3_profit":         [0.05, 0.10, 0.15],
    "force_months":         [9, 10, 12],
    "tp1_close_pct":        [0.50, 0.67, 1.00],
    "tp2_close_pct":        [0.67, 1.00],

    # Mode A
    "macd_fast":            [8, 10, 12, 16],
    "macd_slow":            [24, 26, 28, 30],
    "macd_sig":             [7, 9, 12],
    "div_lookback":         [10, 15, 20, 25],
    "div_min_gap":          [3, 5, 7],
    "vix_ma":               [10, 20, 30],
    "neg_hist":             [True, False],

    # Mode B (ma_short/ma_mid/ma_long fixed at 5/20/50, not searched in v1)
    "pullback_lookback":    [3, 5, 7],
    "touch_tolerance":      [0.0, 0.005, 0.01],
    "hist_converge_days":   [2, 3, 4],
    "iv_rank_max":          [0.30, 0.40, 0.50, 1.00],   # 1.00 = 关闭 IV 过滤（验证其必要性）
}

# 第一性原理约束（无效组合直接跳过）：
# tier1_months < tier2_months < tier3_months ≤ force_months
# macd_fast < macd_slow（且差值 ≥ 16，避免快慢线过近产生噪音信号）
# div_min_gap < div_lookback / 2
# lot_pct < lot_pct_max
```

---

## 综合评分函数

沿用 QQQ 版权重（如 CAGR 与 Max DD 出现明显 tradeoff 再调整）：

```python
WEIGHTS = {"cagr": 0.40, "neg_dd": 0.30, "sharpe": 0.20, "calmar": 0.10}

def composite(r: dict) -> float:
    return (
        WEIGHTS["cagr"]   *  r["cagr"]
        + WEIGHTS["neg_dd"] * (-r["max_dd"])
        + WEIGHTS["sharpe"] *  r["sharpe"] / 5.0
        + WEIGHTS["calmar"] *  r["calmar"] / 10.0
    )
```

---

## Auto-Research 协议

### 运行流程

```bash
# 1. 下载数据
uv run backtest.py --refresh

# 2. 三种模式基线（默认参数，训练集）
uv run backtest.py --mode A  --start 2015-01-01 --end 2024-12-31
uv run backtest.py --mode B  --start 2015-01-01 --end 2024-12-31
uv run backtest.py --mode AB --start 2015-01-01 --end 2024-12-31

# 3. 随机搜索（各模式独立搜索）
uv run optimize.py --mode A  --n 300 --train-start 2015-01-01 --train-end 2024-12-31 --out best_A.json
uv run optimize.py --mode B  --n 300 --train-start 2015-01-01 --train-end 2024-12-31 --out best_B.json
uv run optimize.py --mode AB --n 300 --train-start 2015-01-01 --train-end 2024-12-31 --out best_AB.json

# 4. OOS 测试（2025–今）
uv run backtest.py --mode A  --params best_A.json  --start 2025-01-01
uv run backtest.py --mode B  --params best_B.json  --start 2025-01-01
uv run backtest.py --mode AB --params best_AB.json --start 2025-01-01
```

### Claude 分析循环

每轮 optimize 后：

1. **读取 Top 5 结果**：参数一致 = 强偏好可收窄；参数散乱 = 不敏感可固定
2. **模式归因（AB 专属）**：按 signal_mode 拆分交易，A、B 各自的胜率/平均收益/持有期——若某模式明显拖后腿，考虑单独运行另一模式
3. **验证过滤器价值**：`earnings_blackout=0` 与 `iv_rank_max=1.00` 的组合是否进入 Top 5？若关闭过滤反而更好，说明该过滤器无效或有害
4. **检查训练/OOS 差异**：差值 > 20% CAGR → 过拟合，简化参数空间
5. **关注信号频率**：健康范围 30–100 笔 / 10 年；Mode A 预期偏稀有（个股底背离+大盘恐慌共振少），Mode B 预期偏频繁——AB 组合频率过高时优先收紧 B

### 关键指标解读

| 指标 | 健康范围 | 警惕信号 |
|------|---------|---------|
| CAGR（训练） | 15%–40% | > 50%（可能过拟合） |
| Max DD | < 30% | > 50%（心理承受极限） |
| Sharpe | > 0.8 | < 0.5 |
| Win Rate | 60%–85% | < 50% |
| n_trades | 30–100 / 10 年 | < 15 或 > 150 |
| 训练 vs OOS CAGR 差值 | < 10% | > 20%（明显过拟合） |

---

## 第一性原理约束

设计或修改策略时，始终追问：

- **Mode A 的 histogram 必须为负** — 才是空头区域的底背离，而非多头区域的噪音
- **Mode B 的 MA20 > MA50 必须先行** — 否则「回踩 MA20」在下跌趋势中是接飞刀
- **A 和 B 的波动率条件方向相反是设计而非矛盾** — A 买恐慌（贵但低），B 买平静（便宜但不是最低）；合并为 AND 会互相抵消
- **财报静默期是否过滤掉了最好的信号？** 检查历史上财报后暴跌（如 2019-01 业绩预警）的入场机会是否被误伤——blackout 只应挡住财报**前**的入场
- **tier 时间严格递增** — tier1 < tier2 < tier3 ≤ force
- **FIFO 轮动 = 自动展期** — 用新的低价合约替换旧合约，是设计取舍

---

## 首次基线结果（默认参数，未调优，仅供 sanity check）

训练期 2015-01-01 ~ 2024-12-31，起始资金 $100,000：

| Mode | CAGR | Max DD | Sharpe | Win Rate | # Trades |
|------|------|--------|--------|----------|----------|
| A（底背离反转） | 12.5% | 64.2% | 0.40 | 71.0% | 107 |
| B（趋势回调续涨） | 25.8% | 32.0% | 0.73 | 93.0% | 71 |
| AB（组合） | 39.2% | 66.3% | 0.74 | 79.8% | 178（A 107 / B 71，win_rate 与 avg_pnl 各自匹配单模式结果） |

**注意：这些是默认参数下的原始结果，尚未跑 auto-research 搜索，不代表策略的真实上限或下限。** 观察：
- Mode B 默认参数下 Max DD 明显低于 Mode A（32% vs 64%），初步支持「低 IV 环境买回调」比「高波动环境买反转」风险更低的假设，但样本量小（71 笔）需要参数搜索验证是否稳健
- Mode A 的 64% 回撤远超「已知局限」中列出的健康范围（<30%），是调参的首要目标
- AB 组合的 66.3% 回撤由 Mode A 主导，说明简单 OR 合并不会自动继承 B 的低回撤特性——两者共享同一 NAV 部署上限，A 的糟糕仓位会拖累整体

下一步：跑 `optimize.py --mode A`、`--mode B`、`--mode AB` 各 300 次搜索，看 Max DD 能否被压缩到健康区间。

---

## 已知局限与取舍

1. **无历史 IV 数据**：实现波动率代替 IV，个股的波动率风险溢价比指数更大，回测收益高估幅度也更大
2. **财报 IV 动态无法建模**：BSM + 实现波动率无法体现财报前 IV 拉升/财报后 crush；静默期只是粗糙缓解
3. **无价差/滑点**：个股期权价差通常比 QQQ 宽
4. **整数合约忽略**：回测允许小数份额，实盘 1 合约 = 100 股
5. **Mode B 是未验证的新设计**：QQQ 版只验证过 Mode A 的逻辑；B 的所有参数默认值均为初始猜测，依赖 auto-research 校准

---

## 未来工作（明确不进 v1）

- **POC / Volume Profile 筹码支撑**（Gemini 建议第 4 项）：需要 volume-profile 近似算法，日线 OHLCV 无法直接计算；待 A/B 模式验证后评估
- **MA5/MA20 死叉提前出场**：见出场规则「候选增强」
- **扩展到 MSFT、GOOGL**：AAPL 版参数验证后，同一框架换 ticker 重跑，检验参数可迁移性
- **RSI 触底反弹**（Gemini 建议）：与 MACD 底背离高度相关，v1 不重复引入；若 Mode A 信号过稀可作为放宽条件测试
