2、波士顿房价预测数据集
格式：NumPy二进制格式.npy文件(numpy.ndarray)，N维数组形式
使用方法：import NumPy模块后，var = numpy.load("xxx.npy")导入
命名方式：数据类型+数据集划分类型.npy
[1]数据类型
x：特征数据
y：目标数据

[2]数据集划分类型
train：训练集(70%)
test：测试集(30%)
无标签：完整数据集(100%)

boston.csv：未经预处理的原始数据集文件，来源http://lib.stat.cmu.edu/datasets/boston
