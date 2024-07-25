笔记及代码根据PenicillinLP大佬视频 [【强化学习】一小时完全入门](https://www.bilibili.com/video/BV13a4y1J7bw/?spm_id_from=333.337.search-card.all.click) 修改，特此感谢！


# 强化学习

强化学习是Agent 为了在与环境的互动中达成特定目标而进行的学习过程

## 基本元素

- Agent 智能体

- Environment 环境

- Goal 目标

## 主要元素

- State 状态

- Action 行动
- Reward 奖励

## 核心元素

- Policy 策略
- Value 价值
  - 未来能获得奖励的期望值

## 特点

- 试错学习

- 延迟奖励

## 核心问题

- 权衡开发和探索 Exploitation&Exploration

## 算法

### 多臂老虎机

学习行动具有的价值

#### 价值函数：

采样平均，选择a获得的奖励除以选择a的次数
$$
Q_t(a) = \frac{\sum_{i=1}^{t-1}R_i*1(A_i=a)}{\sum_{i=1}^{t-1}1(A_i=a)}, a={L}/{R}
$$
第n次实际获得的奖励-预测获得的奖励，奖励分布不随时间变换，随着采样的次数的增加学习率减小
$$
Q_{n+1} = Q_n + \frac{1}{n}(R_n - Q_n)=\frac{1}{n}\sum_{i=1}^nR_i
$$

奖励分布随时间变换，不希望学习率减小，不同时间获得的奖励权重不同
$$
Q_{n+1} = Q_n + \alpha (R_n - Q_n)
= (1-a)^nQ_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i}R_i
$$


$$
Q(s, a) \leftarrow Q(s, a) + \alpha \times (r + \gamma \max_{a'} Q(s', a') - Q(s, a))
$$

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \times (Q(s', a') - Q(s, a))
$$

#### 策略函数：

$$
A_t = argmax_a Q_t(a)
$$
