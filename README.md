# MyKBQA
这是一个KBQA的demo,模型是key-value Memory Net 

遇到的一个问题的解决方案（2019.3.23）  

Q:一个问题很大程度上会出现多个答案，而一些网上的做法是将问题分成多份  

举个例子：  

Q->A1,A2,A3   

转化成三个问题  

Q->A1  

Q->A2  

Q->A3  

这样会遇到一些问题，就是进行神经网络输入的时候，输入的问题是一样的，但是答案却不一样，例如问题“张三的儿子是谁？” 假设问题的答案有两个，一个是张小明，一个是张大明，但是，如果前一次训练刚刚将一些参数训练地调整到偏向“张晓明”，但是下一次训练又将参数往"张大明"的方向调整，这样的带有二义性的训练是不可能训练处一个好的模型的，下面给出我的一个解决方案（自己想的）  

Solution：  

再进行转化问题的时候，即  

Q->A1  

Q->A2  

Q->A3  

这一步上，给问题加一个编号  

#1 Q->A1  

#2 Q->A2  

#3 Q->A3  

编号需要预先添加到idx索引文件中，方便可以查到编码，至于加多少，查找QA数据集中答案的最多个数
同时，为了将编号和问题尽量进行一定的关联，设置一些关联规则，比如将问题按照首字母排序（汉字按照拼音顺序）
这样，在进行测试集进行测试的时候，也是尽可能的先将测试集中的答案进行字典排序
然后，将问题按照编号进行划分成多份进行预测
这样就能在一定程度上去除掉二义性
或者直接将问题的数据集进行原子级的划分，用编号来区分

然后，在接收用户输入的时候，预先在问题的前面加上编号，来尽可能多的实现答案的完整性，加多少编号，就会返回多少个答案，可能会有重复，这是所谓的Top-K

-------------------------------------------------------------------------------------------------------------------------------------------
（2019.3.26更新，训练集效果显著，测试集有一些问题）
![image](https://github.com/Joseph1314/MyKBQA/blob/master/%E8%AE%AD%E7%BB%83%E9%9B%86%E6%95%88%E6%9E%9C.jpg)

----------------------------------------------------------------------------------------------------------
(2019.3.27更新，训练集准确率接近1，测试集0.01(尴尬.jpg) ）  
数据是使用的小批量的训练（大约只有10k的数据量，打算加入一些防止过拟合的方法来看看效果  

----------------------------------------------------------------------------------------------------------------------------------------
(2019.3.28 更新)
当前遇到的一些问题以及接下来的思路方案：
1.训练的时候，需要学习词嵌入矩阵Embedding，这就要求训练的时候，能够尽可能地把一些实体覆盖，这样才能尽可能地学习到每一个词的embedding的向量表示  
这里就会存在一个冲突，必须要给神经网络喂大量的数据才能够学习到较好的词嵌入表示->进而在测试集上能有良好的表现，这是已经通过实验验证了的（数据集的增大会一点点提高测试集的准确率），但是，随着训练集的增大，训练的时间花销变得越来越大，训练会变得越来越慢，进而调试的时间成本变高，因为训练周期特长，所以修改一些超参数之后，能看到修改效果的时间间隔就特别长。

2.因为每个问题都可能对应多个答案，所以我之前给出的一个解决方案是，将一对多的Q-A，转化成多对多，并且给每个问题之间加一个编号"#i" 来进行唯一性的区分，并且预处理的时候，将答案进行了一个排序，这样编号小的问题会对应字典序较小的答案。
后来我感觉这样尽管可能会在一定程度上减少问题的答案之间的冲突，但是每次就会使得批处理的数据大小不确定，可能并不会很好的训练一个比较合适的模型，

所以我想尝试下面的一个想法：

回到最初，修改损失函数的表示方法，自己编写一个自定义的损失函数，这样一条Q-A还是一条数据

自定义的损失函数的思路是这样的：

现将out输出做一个softmax，然后用TensorFlow的top_K函数选择k个候选答案的索引index，k值的选择应该和一条QA数据集中的答案个数相同

然后可以进行一个sort排序，和答案中的answer对应，（因为答案也是事前排好序的）

然后设计一个loss函数，可以是上面的index和answer对应位置的数的reduce_mean均方根，这样就能实现多个答案并存的问题，还能扩大命中的准确率，毕竟命中多个答案中的一个，比命中唯一个答案要容易，哪怕是top_k中只有一个命中了答案，也是好的，这样尽管不全面，但是至少能提高准确率。

