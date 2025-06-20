#### 网格信息

- forbidden area：(1,1) (2,3) (3,2)，Reward = -0.5
- Terminal：(4,4)，Reward = 1.0
- normal：Reward = 0.0
- GAMMA = 0.9
- THETA = -1e9

#### 04 - 值迭代

- 初始化状态值V（比如全设为0），定义一个策略并赋初值（赋予多少不重要，仅仅为定义变量赋初值）
- 使用贝尔曼公式反复更新每个状态的最大V,直到收敛，然后再从使用最大V求出最优动作（策略）

![image-20250620081124085](https://raw.githubusercontent.com/sleepyDev0x/Pictures/main/52bd1cc3f8a1468f96d39aad34eddbc4.png)	

#### 04 - 策略迭代

- 给定一个初始策略（比如action全是向上），状态值初值设为0（设为几不重要，仅仅是定义这个变量赋予一个初值）
- 策略评估：评估在这样一个策略下，对于所有(s,a)的状态值，这个过程实际上就是求贝尔曼方程的过程，我们使用迭代法去求，所以引入delta就是为了判别v是否收敛为状态值
- 策略改进：拿到在评估阶段求得的状态值V，根据贝尔曼公式去求动作值（即时奖励+gamma*未来回报），选择动作值最大的action，更新策略

![image-20250620081220667](https://raw.githubusercontent.com/sleepyDev0x/Pictures/main/c45d7433a37086177883a0685c54e6a2.png)	