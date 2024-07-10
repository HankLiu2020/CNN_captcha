trainRoot = "./训练数据集"
testRoot = "./测试数据集"

# 不可修改的参数
tensorLength = 248
charLength = 62
charNumber = 4
ImageWidth = 32
ImageHeight = 32

# 可修改的参数
learningRate = 1e-3
totalEpoch = 20
batchSize = 512
printCircle = 400
testCircle = 800#0
testNum = 6 #test_batch_num
saveCircle = 800#200

#10w个样本，一轮500，要200batch才能遍历一回