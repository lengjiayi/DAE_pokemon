一个学习用的Deep Auto-Encoder模型，使用了李宏毅老师提供的宝可梦数据集

- 共有五层编码器和五层解码器。

- 训练时采用BCE误差，Adam优化器

### 重构测试

编码-解码效果如下：

![](https://github.com/lengjiayi/DAE_pokemon/blob/master/assets/ATE.PNG)

### 生成测试

我尝试使用DAE产生一些中间图片。

之前听说过DAE无法很好的产生新的数据，这是由于训练时无法保证编码空间里样本点周围的点都可以有效解码，且和样本相关联。

我选择了比较相似的宝可梦之间的向量作为生成码，具体即将两个样本之间的高维空间线段等分后进行解码。

![](https://github.com/lengjiayi/DAE_pokemon/blob/master/assets/DAE_GEN.PNG)

_./gif文件夹中存有生成过程的gif动态图_

