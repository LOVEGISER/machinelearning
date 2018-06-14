from  numpy  import  *
#随机产生4*4 矩阵
x = random.rand(4, 4)
#print(x)
#数组转换矩阵
randMat =mat(x)
print(randMat)
#矩阵逆运算
tranx = randMat.I
print(tranx)