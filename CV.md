## MTCNN
1. 结构
  1. Pnet
  2. Rnet
  3. Onet

2. 训练

3. NMS：

由于我们采用的都是滑动窗口产生的候选框，所以我们需要选择一个最适合的框。所以我们要做如下的操作：

	1.选择当前得分最高的框
	2.遍历其余框，如果IOU大于threshold的话，我们将它删除。
	3.从未处理的框中选择一个得分最高的，重新做这个步骤。

code
```
def cpu_nms(dets, thresh):
	x1 = dets[:, 0]
	x2 = dets[:, 1]
	y1 = dets[:, 2]
	y2 = dets[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	score = dets[:, 4]
	order = scores.argsort()[:,:,-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]]) 
		yy1 = np.maximum(y1[i], y1[order[1:]]) 
		xx2 = np.minimum(x2[i], x2[order[1:]]) 
		yy2 = np.minimum(y2[i], y2[order[1:]]) 
		#计算相交的面积,不重叠时面积为0 
		w = np.maximum(0.0, xx2 - xx1 + 1) 
		h = np.maximum(0.0, yy2 - yy1 + 1) 
		inter = w * h #计算IoU：重叠面积 /（面积1+面积2-重叠面积） 
		ovr = inter / (areas[i] + areas[order[1:]] - inter) 
		#保留IoU小于阈值的box
		inds = np.where(ovr <= thresh)[0] 
		order = order[inds + 1] 
```

4. 人脸对齐


## Inception_ResNet_V1:
1. ResNet

  一直持续增加深度，无法克服梯度弥散的问题。
  需要使用一个跳过连接的形式，可以一定程度上避免这样的问题。解决梯度弥散之后，更深的网络可以提取更多的特征。就可以达到更好的效果。

2. Inception

  网络已经足够深，我们不能从深度上做文章。考虑能不能使同时使用不同size的kernel，去同时采集不同的信息，之后做一个结合操作。这就是inception的初衷，我们同时使用了1*1，3*3 和5*5的卷积核，然后把每一个卷积核得到的feature map结合得到最终的feature map。但是，卷积核从3*3到5*5，计算成本会大大增加。所以，我们采用了1*1的卷积核，去减少了参数的数量和运算的次数。达到一个优化的效果。


这里我们用的网络是他们结合之后的Inception_ResNet_v1:

![architecture](https://raw.githubusercontent.com/liuyaqiao/Learning-Note/master/inception_resnetv1.png)

可以看到主要由reduction block和inception_resnet block组成。

3. KCF追踪（核相关滤波算法）：

一般的追踪问题：

1. 在It帧中，在当前位置pt附近采样，训练一个回归器。这个回归器能计算一个小窗口采样的响应。 
2. 在It+1帧中，在前一帧位置pt附近采样，用前述回归器判断每个采样的响应。 
3. 响应最强的采样作为本帧位置pt+1。

判别式跟踪，主要是使用给出的样本去训练一个判别分类器，判断跟踪到的是目标还是周围的背景信息。主要使用轮转矩阵对样本进行采集，使用快速傅里叶变化对算法进行加速计算

`轮转矩阵`, `判别式分类器`, `傅立叶变换`,`核方法`

好处：
1. 思路简单，代码开源
2. 速度快（100+fps）

缺点：
1. 多目标追踪不理想
2. 形变比较大的时候不理想


## 人脸识别优化

1. 
  我们把mtcnn得到的结果坐标，传入KCF追踪器。并设立了一个时间戳，使得检测算法和追踪算法交替进行。得到的结果是：速度变快，并且准确率提高；因为MTCNN这个model经常会出现错误，会把眼镜当作人脸出现误检；我们的检测追踪结合的检测器基本不会出现这样的问题；

测试：
	1. 速度快
	2. 准确率提高：因为我们这个是面向工厂，所以没有在一些官方数据集上跑。
	我们自己生成的视频：去数每一帧检测的结果；提高了检测的结果；
	
2. 传统方法优化：
  一帧图像很容易出现误检的情况，所以我们结合多张图像的信息；

3. 改变feature map
  采集了左脸、中脸、右脸作为特征储存在了得到的人脸的数据库中；使得模型更加鲁棒；具体的做法是：我们把输入变成了视频输入，通过dlib的一函数，检测出当前是左脸还是右脸，然后存三个feature，虽然查找时间变长，但是增强了准确性。

4. 工程方面：


## HOG
1. 主要思想
  一副图像中，局部目标的表象和形状能够被梯度或边缘的方向密度分布很好的描述。（本质是梯度的信息，而梯度主要存在与边缘的地方）。

2. 具体做法：
  把图像分成小的联通区域，我们称为细胞单元。然后采集细胞单元中各个像素点的梯度或者边缘的方向直方图，最后把这些直方图组合起来，可以构成特征描述器。

3. 优化：
  把这些局部直方图在图像的更大的范围内（我们把它叫区间或block）进行对比度归一化（contrast-normalized），所采用的方法是：先计算各直方图在这个区间（block）中的密度，然后根据这个密度对区间中的各个细胞单元做归一化。通过这个归一化后，能对光照变化和阴影获得更好的效果。

4. 优点：
  与其他的特征描述方法相比，HOG有很多优点。首先，由于HOG是在图像的局部方格单元上操作，所以它对图像几何的和光学的形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上。其次，在粗的空域抽样、精细的方向抽样以及较强的局部光学归一化等条件下，只要行人大体上能够保持直立的姿势，可以容许行人有一些细微的肢体动作，这些细微的动作可以被忽略而不影响检测效果。因此HOG特征是特别适合于做图像中的人体检测的。

但是速度比较慢，对噪点敏感。

5. 实现

1）灰度化
2）gamma矫正，对颜色空间标准化，降低了局部阴影和光照影响，较少了噪声。
3）计算梯度：每一个像素的梯度的大小和方向
4）讲图像分为小cells（6*6）
5）统计每一个cell的直方图，可以形成一个cell的描述子
6）将每几个cell组成一个block，每一个block的desciptor串联起来就是该block的HOG特征描述子，对block算子进行标准化，得到一个整体描述子，这里block之间会有重叠，为了获得相邻像素之间的关系。之后对一个block中的元素进行标准化。
7）将所有的block的描述子串联起来，得到整个的HOG特征描述子

## MPI：

它是一种基于消息传递的接口。MPI程序是基于消息传递的并行程序，消息的传递是并行执行的进程具有自己独立的堆栈和代码段。作为互不相关的多个程序独立执行，进程之间的信息交互通过显示地调用通信函数来完成。

我们这里用到的是一个管理者，多个工人的MPMD程序；

## kmeans：
把一堆数据分为K类。
1. 随机选择k个分类中心；
2. 对于每一个样本，计算它所属于的类（里中心点最近的距离）
3. 对于每一个类，重新计算中心点
4. 如果不收敛，重复2和3的操作；

```
def initCentroids(dataset, k):
    dataSet = list(dataSet)
    return random.sample(dataSet, k)
	
def minDistance(dataSet, centroidList):
    clusterDict = dict() #dict保存簇类结果
    k = len(centroidList)
    for item in dataSet:
        vec1 = item
        flag = -1
        minDis = float("inf") # 初始化为最大值
        for i in range(k):
            vec2 = centroidList[i]
            distance = calcuDistance(vec1, vec2)  # error
            if distance < minDis:
                minDis = distance
                flag = i  # 循环结束时， flag保存与当前item最近的蔟标记
        if flag not in clusterDict.keys():
            clusterDict.setdefault(flag, [])
        clusterDict[flag].append(item)  #加入相应的类别中
    return clusterDict  #不同的类别

def getNewCentroids(clusterDict):
	centroidList = []
    for key in clusterDict.keys():
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList  #得到新的质心

def getVar(centroidList, clusterDict):
    # 计算各蔟集合间的均方误差
    # 将蔟类中各个向量与质心的距离累加求和
    sum = 0.0
    for key in clusterDict.keys():
        vec1 = centroidList[key]
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = item
            distance += calcuDistance(vec1, vec2)
        sum += distance
    return sum

def test_k_means():
    dataSet = loadDataSet()
    centroidList = initCentroids(dataSet, 4)
    clusterDict = minDistance(dataSet, centroidList)
    # # getCentroids(clusterDict)
    # showCluster(centroidList, clusterDict)
    newVar = getVar(centroidList, clusterDict)
    oldVar = 1  # 当两次聚类的误差小于某个值是，说明质心基本确定。

    times = 2
    while abs(newVar - oldVar) >= 0.00001:
        centroidList = getCentroids(clusterDict)
        clusterDict = minDistance(dataSet, centroidList)
        oldVar = newVar
        newVar = getVar(centroidList, clusterDict)
        times += 1
        showCluster(centroidList, clusterDict)

```

kmeans：
好处：
1. 简答容易实现
2. 可解释性比较强，也比较容易并行
3. 聚类效果比较好，调参数比较简单，收敛快

坏处：

1. k的取值不好取，而且需要提前确定
2. 局部最优，和初始化有关
3. 只能发现球型簇
4. 对孤立点和离群点比较敏感
5. 不能处理非convex分布

　　并行化思路：

　　使用主从模式。由一个节点充当主节点负责数据的划分与分配，其他节点完成本地数据的计算，并将结果返回给主节点。大致过程如下：

　　1、进程0为主节点，先从文件中读取数据集，然后将数据集划分并传给其他进程；

　　2、进程0选择每个聚类的中心点，并发送给其他进程；

　　3、其他进程计算数据块中每个点到中心点的距离，然后标出每个点所属的聚类，并计算每个聚类所有点到其中心点的距离之和，最后将这些结果返回给进程0；

　　4、进程0计算出新的中心点并发送给其他进程，并计算其他进程传来的聚类所有点到其中心点的距离总和；

　　5、重复3和4直到，直到步骤4中的所有聚类的距离之和不变（即收敛）。

master:
1. 读取数据
2. 随机选择聚类中心
3. 分配数据给workers
4. 计算出新的中心
5. 判断是否收敛

workers：
1. 接受数据和中心点
2. 计算最近的距离，和所属的cluster，将每个节点的label传回和sum传回给中心点

照片
500 * 500， 750 * 750， 1000*1000， 1250 * 1250的照片，并且选了k（3，5，20， 50）

在小照片和k小的情况下，加速效果不明显。当k = 50的时候，会有比较大的speedup；

## svm做车辆检测：

1. LinearSVC：
  线性核，调整了惩罚项C，使用了hinge loss和l2 loss；
  validation：

2. dataset：

4k 车和非车的图片-->一部分

分成验证集和测试集：

3. 提取特征：
  1. hog
  2. color hist -> bins = 32

4. sliding windows(64*64) -> 有overlap

5. 问题：
  错检测；一个车有很多框；

6. heatmap：
  连续检验多帧，如果超过阈值判断为有；不超过判断为没有；
  这里也是类似与NMS的思路；

SVM + HOG的优缺点：

优点：

1. 检测效果比较好，误检率小；


缺点：
1. 很难处理遮挡问题
2. 不具有旋转不变性
3. 对噪点敏感

总体来说：

基于滑动窗口的选取没有策略性，时间复杂度高，窗口冗余。
手工设计的特征对于多样性的变化没有很好的鲁棒性



## Spark：

是一个内存计算的大数据开源集群计算环境；不会写入磁盘中，一直缓存在内存中。

Spark 的主要特点还包括:
- (1)提供 Cache 机制来支持需要反复迭代计算或者多次数据共享,减少数据读取的 IO 开销;
- (2)提供了一套支持 DAG 图的分布式并行计算的编程框架,减少多次计算之间中间结果写到 Hdfs 的开销;
- (3)使用多线程池模型减少 Task 启动开稍, shuffle 过程中避免不必要的 sort 操作并减少磁盘 IO 操作。(Hadoop 的 Map 和 reduce 之间的 shuffle 需要 sort)

### RDD
1. 数据储存在多个机器上，但是在一个RDD中；
2. RDD有一些操作是Action和Transformation； 失败自动重建
3. DAG -> 有向无环图 -> 容错
4. shuffle、stage
  shuffle：对数据进行重组，是map和reduce的桥梁；
5. cache
6. map_reduce

## GPU

### 架构：ALU heavy，高度并行，大吞吐量

cpu ： cache heavy, focus on individual thread

### 程序：
1. copy data（分配空间，拷贝数据）
2. invoke kernel
3. copy back

grid -> 一个kernel
block -> thread的集合

## 网络
传输层：
TCP： 可靠的传输层通信协议；
UDP：提供了不可靠的信息传送服务；不建立连接，不需要进行维护连接状态；快；尽最大努力交付，但是不可靠；不保证顺序；
IP：网络层的协议

三次握手：所谓三次握手(Three-way Handshake)，是指建立一个 TCP 连接时，需要客户端和服务器总共发送3个包。

进程线程：

进程：进程是表示资源分配的基本单位，又是调度运行的基本单位

线程：线程是进程中执行运算的最小单位，亦即执行处理机调度的基本单位。如果把进程理解为在逻辑上操作系统所完成的任务，那么线程表示完成该任务的许多可能的子任务之一。

进程有独立的地址空间，一个进程崩溃后，其他进程继续执行；但是线程没有，线程只有自己的stack pointer，一个线程死掉进程也会死掉；

1. 线程小
2. 进程有独立的内存，多线程共享内存
3. 线程不能独立执行，需要依赖进程

进程之间的通信方式IPC：管道、共享内存、套接字、信号量、消息排队

线程之间的通信方式：锁机制、信号量机制、信号机制

进程是资源分配的基本单位，有独立的地址空间。
线程是资源调度运行的最小单位。线程是拥有自己的局部变量和栈，线程之间共享堆。

## c和python
python是解释性语言，c是编译性语言，可以转换成汇编，机器语言，更快。不存在动态类型，动态检查。

而python是逐句解释执行的。在运行是才会解释；

## sigmoid的缺点：
导数小，容易梯度消失；只能解决二分问题；

## SVM通过松弛变量控制过拟合；

什么样的函数可以称为kernel函数：





## kmeans 如果确定k：

## SVM对异常点不敏感，因为它只和sv有关系；而LR敏感；SVM更加健壮；

## L1选择特征：把不重要的特征压缩成0

## c++基础
1. 析构函数
2. 构造函数
3. 虚函数

## 不均衡数据分类
1. 扩大数据集
2. 其他评价方法
3. recall precision, F1, ROC curve, 调整分类阈值；
4. 重新采样
5. 人工增加样本
6. 不同的算法：决策树
7. 集成学习 random forest

## LR并行：





## 深度学习

1. BP算法

2. GRU和LSTM的图，作用

   LSTM 通过引入cell state、input gate、forget gate和outputgate的方式实现长期记忆； cell state: 表示前些时刻的信息; 相当于小本子，用来记忆的东西；每次多储存了一些参数和状态，来完成long term memory。 forget gate: 决定丢弃cell中的哪一部分 input gate: 决定什么样的新信息课一加入到cell中 output gate: 更新细胞状态后的h层输出

   LSTM用加和的方式替代了乘积，不容易出现梯度弥散。但是有可能出现梯度爆炸。

3. pooling作用，max pooling和average pooling分别的使用范围
  引入不变性（平移不变，旋转不变，尺度不变）；减少下一层的尺寸和参数数量；获得一个固定长度的输出；防止过拟合。
  max-pooling为了取得图片最为显著的特征；

average pooling -> resnet 最后一层
平均特征会使图片变得更加平滑；global average pooling；防止过拟合不用fc；

4. 梯度消失和梯度爆炸的原因和解决方法：
  梯度爆炸：误差梯度在网络训练时被用来得到网络参数更新的方向和幅度，进而在正确的方向上以合适的幅度更新网络参数。在深层网络或递归神经网络中，误差梯度在更新中累积得到一个非常大的梯度，这样的梯度会大幅度更新网络参数，进而导致网络不稳定。

  问题：无法更新权重，误差函数不收敛；出现NAN
  解决：CNN，RNN
  1. 重新设计网络结构，用性质比较好的激活函数。
  2. 在RNN中用LSTM，GRU
  3. 初始化权重，BN，xavier init
  4. 残差

5. LSTM如何解决梯度消失和梯度爆炸？
  梯度爆炸不是一个很严重的问题，可以根据裁剪后的优化算法来解决；它是通过加法来解决梯度消失；

6. 防止过拟合的方法：

7. 越大的batch_size得到的收敛速度越快，但会造成内存的溢出；

8. dropout：
  思想其实是集成了bagging的思想，每一次训练都随机使一些神经元失活；它可以起到一个防止过拟合的过程；

9. 优化方法的联系：sgd, momentum, rmsprop, adam

  sgd最普通的梯度下降
  momentum：考虑了之前前一步的影响，把前一步的下降方向也加入到了新的下降计算中；
  rmsprop：在引入动量的同时也改变了学习率
  adam对rmsprop加入了修正

10. 1*1卷积核的作用：
  改变channel的维度；减少计算成本和参数数量；

11. softmax loss 和 交叉熵：
   softmax分类器用的是交叉熵误差函数的形式；

12. range 返回的是一个数组 xrange 返回的是一个生成器 在大数组的时候，xrange更好，不需要一开始就分配大空间。

13. model
   Lenet 1st
   Alexnet： Relu dropout
   VGGNet: 1*1 3*3
   inception Net: 去除了fc，用average pooling去替代；inception module；V2->BN
   ResNet: residual block

   mobile net 通过depthwise的操作减少了参数的个数；

14. 机器学习优化方法：

1阶：梯度下降；随机梯度下降。。。
2阶：牛顿法：牛顿法的基本思想是利用迭代点![x_k](http://latex.codecogs.com/gif.latex?x_k)处的一阶导数(梯度)和二阶导数(Hessen矩阵)对目标函数进行二次函数近似，然后把二次模型的极小点作为新的迭代点，并不断重复这一过程，直至求得满足精度的近似极小值。

-> 收敛速度快，但是复杂度高



15. 当内容具有局部相关性；并且由低层次的特征经过组合，组成高层次的特征，并且得到不同特征之间的空间相关性

16. 池化层如何反向传播：记下最大值的位置，其余都写0；如果是平均则和等于该值；





## ML

### 集成学习



bagging：可以并行的训练出不同关系不大的分类起；然后通过投票或其他方法结合；

stacking: 通过初级学习器生成一个训练集，把这个训练集作为次级训练器的输入；

boosting:  可以将一类弱学习器转换为强学习器。必须要串行生成。一般要满足加法模型和前向分布的形式。

根据损失函数的样子，可以分成不同的算法；adaboost是以指数性损失函数来更新的算法。



RF是根据bagging基础上生成的，它同时还引入了随机特征的选择。



GBDT：

1.原理

2.影响因素

3.适用场合



