# AlphaRice
使用AlphaZero的强化学习算法下米字棋

## 什么是米字棋

米字棋是口字中间将横竖斜对角链接，形成米字而得名。

A B 两人分别赤三子，下子阶段可以在九宫格坐标上任意下子，三子下完后进入移子阶段，移子规则是只能沿着米字棋盘的线上移动一个子。

谁先将三个字连成一条线，谁就赢。

## 人机对战

clone 代码后，可以直接运行 `human.py` 文件，开始人机对战。

下子阶段输入形如 `0,1` 的字符串，其中 `0` 表示横坐标为 0，`1` 表示纵坐标为 1；
走子阶段输入形如 `0,1->1,1` 的字符串，表示坐标为 `0,1` 的子移动到坐标 `1,1` 上；

## 编码规则

与其他棋类的强化学习实现不同的地方在于该棋盘如何编码，以及动作空间如何编码的问题。

### 棋盘编码

该棋盘为三乘三的矩阵，为了方便记录每个坐标的子是哪个下棋者的，所以我将三乘三的矩阵横向排列成 (1, 9) 的列表。
根据坐标与该列表索引的映射来相互转换。

### 动作空间

该棋的动作空间有两个阶段，下子阶段和走子阶段，所以我将动作空间划分为两个部分，下子阶段表示为 `01` ，走子阶段类似表示为 `0102`，表示坐标在 0,1 的子移动到 02 的坐标上。



