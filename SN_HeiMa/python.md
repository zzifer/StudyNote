# 第一阶段 第二章 基础语法
## 注释
![[Pasted image 20230107125844.png]]

## 数据类型
![[Pasted image 20230107130646.png]]

## 数据类型转换
![[Pasted image 20230107130845.png]]

## 命名规范
![[Pasted image 20230107132119.png]]

## 运算符
![[Pasted image 20230107132412.png]]

## 字符串的定义方式
![[Pasted image 20230107132751.png]]

## 字符串拼接
![[Pasted image 20230107133017.png]]

## 字符串格式化
![[Pasted image 20230107133158.png]]
![[Pasted image 20230107133230.png]]
![[Pasted image 20230107133333.png]]

## 字符串格式化的精度控制
![[Pasted image 20230107133459.png]]

## 字符串格式化2
![[Pasted image 20230107133655.png]]
![[Pasted image 20230107133725.png]]

## 数据输入input
![[Pasted image 20230107133932.png]]
![[Pasted image 20230107134011.png]]
![[Pasted image 20230107134053.png]]


# 第一阶段 第三章 判断
## if
![[Pasted image 20230108124422.png]]
![[Pasted image 20230108124631.png]]
![[Pasted image 20230108125422.png]]


# 第一阶段 第四章 循环
 ## while
 ![[Pasted image 20230108130616.png]]

## for
![[Pasted image 20230108132110.png]]
![[Pasted image 20230108132304.png]]
![[Pasted image 20230108132410.png]]

## continue和break
![[Pasted image 20230108132516.png]]
![[Pasted image 20230108132616.png]]


# 第一阶段 第五章 函数
## 函数
![[Pasted image 20230108134459.png]]
![[Pasted image 20230108134803.png]]
![[Pasted image 20230108134838.png]]
![[Pasted image 20230108135050.png]]
![[Pasted image 20230108135245.png]]
![[Pasted image 20230108135541.png]]
![[Pasted image 20230109122820.png]]
![[Pasted image 20230109123022.png]]


# 第一阶段 第六章 数据容器
![[Pasted image 20230109121921.png]]
## 列表
![[Pasted image 20230109122044.png]]
![[Pasted image 20230109122600.png]]
![[Pasted image 20230109123345.png]]
![[Pasted image 20230109123916.png]]
![[Pasted image 20230109124229.png]]
![[Pasted image 20230109124543.png]]
![[Pasted image 20230109124619.png]]
![[Pasted image 20230109124655.png]]
![[Pasted image 20230109124732.png]]

## 元组
![[Pasted image 20230109125809.png]]
![[Pasted image 20230109125909.png]]
![[Pasted image 20230109130218.png]]
定义单个元素的元组后面要加逗号要不然会变成字符串
![[Pasted image 20230109130520.png]]
![[Pasted image 20230109130715.png]]
![[Pasted image 20230109130844.png]]
![[Pasted image 20230109131124.png]]

## 字符串
![[Pasted image 20230109133032.png]]
![[Pasted image 20230109133213.png]]
![[Pasted image 20230109133301.png]]
![[Pasted image 20230109133406.png]]
![[Pasted image 20230109133524.png]]
![[Pasted image 20230109133549.png]]

## 切片
![[Pasted image 20230109134956.png]]
![[Pasted image 20230109135044.png]]

## 集合
![[Pasted image 20230109135331.png]]
![[Pasted image 20230109135427.png]]
![[Pasted image 20230109135727.png]]
![[Pasted image 20230109135749.png]]

## 字典
![[Pasted image 20230109135955.png]]
![[Pasted image 20230109140303.png]]
```python
# 定义嵌套字典
stu = {
	   "张三"：{
		   "语文"：7,
		   "数学"：8
	   },
	   "李四"：{
			"语文":5,
			"数学":4
		}
}

stu["张三"]["语文"]
```
![[Pasted image 20230109140957.png]]

## 总结
![[Pasted image 20230109141114.png]]
![[Pasted image 20230109142434.png]]


# 第一阶段 第七章 函数进阶
## 函数的多返回值
![[Pasted image 20230109142903.png]]

## 函数的多种参数使用形式
![[Pasted image 20230109143300.png]]
![[Pasted image 20230109143422.png]]
![[Pasted image 20230109143539.png]]
![[Pasted image 20230109143856.png]]
![[Pasted image 20230109144026.png]]

## 函数作为参数传递
![[Pasted image 20230109145114.png]]
这里的形参和实参可以不一样

## lambda匿名函数
![[Pasted image 20230109145840.png]]
![[Pasted image 20230109145934.png]]


# 第一阶段 第八章 文件操作
## 文件的读取
![[Pasted image 20230109152046.png]]
![[Pasted image 20230109152136.png]]
![[Pasted image 20230109152326.png]]
![[Pasted image 20230109152727.png]]
![[Pasted image 20230109152906.png]]
![[Pasted image 20230109153110.png]]

## 文件的写出
![[Pasted image 20230109153356.png]]
![[Pasted image 20230109153633.png]]

## 文件的追加写入
![[Pasted image 20230109154213.png]]

# 第一阶段 第九章 异常和模块
## 异常的捕获
```python

try:
	可能发生错误的代码
except:
	如果出现异常执行的代码

# 捕获指定异常
 try:
 except NameError as e:

# 捕获多个异常
try:
except (NameError, ZeroDivisionError) as e:

# 捕获所有异常
try：
except Exception as e:

# 异常else
# else表示的是如果没有异常要执行的代码
# finally不管有没有异常都会执行
try:
except Exception as e:
else:
finally:
```

## 异常的传递
![[Pasted image 20230105220219.png]]

## 模块的导入
![[Pasted image 20230105221646.png]]
![[Pasted image 20230105221955.png]]
![[Pasted image 20230105222059.png]]

当导入多个模块的时候，且模块内有同名功能，当调用这个同名的功能的时候，调用的是后面导入的模块的功能

##  main和all
![[Pasted image 20230105223231.png]]
![[Pasted image 20230105223257.png]]
直接执行会运行这个test(1,2)但是如果被别的模块调用则这段代码不执行
![[Pasted image 20230105223414.png]]

## python包
![[Pasted image 20230105223858.png]]
创建包后，包内会自动创建_init_.py这个文件，这个文件控制包的导入行为

![[Pasted image 20230105224055.png]]
![[Pasted image 20230105224109.png]]

# 第二阶段 第一章 面向对象
## 创建对象
![[Pasted image 20230106110140.png]]

## 类的成员方法
![[Pasted image 20230106130635.png]]
![[Pasted image 20230106130705.png]]

## 构造方法
![[Pasted image 20230106131808.png]]
![[Pasted image 20230106132716.png]]

## 魔术方法
![[Pasted image 20230106133334.png]]
![[Pasted image 20230106133407.png]]
![[Pasted image 20230106133423.png]]
![[Pasted image 20230106133531.png]]

## 封装
![[Pasted image 20230106133936.png]]

## 继承
![[Pasted image 20230106140108.png]]
![[Pasted image 20230106140249.png]]
![[Pasted image 20230106140412.png]]
![[Pasted image 20230106140529.png]]

## 复写
![[Pasted image 20230106140713.png]]
![[Pasted image 20230106140820.png]]

## 类型注解
![[Pasted image 20230106142007.png]]
![[Pasted image 20230106142141.png]]
![[Pasted image 20230106142518.png]]
![[Pasted image 20230106142607.png]]

## union类型
![[Pasted image 20230106145425.png]]
![[Pasted image 20230106145454.png]]

## 多态
![[Pasted image 20230106145711.png]]
![[Pasted image 20230106145807.png]]


# 第三阶段
## 正则表达式
### 基础方法
![[Pasted image 20230109162343.png]]
![[Pasted image 20230109162603.png]]

### 元字符匹配 没看懂
![[Pasted image 20230109165716.png]]
![[Pasted image 20230109170022.png]]
![[Pasted image 20230109170049.png]]