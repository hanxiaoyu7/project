# Pycharm

### 安装中文插件

![](E:\学习笔记\pic\pycharm1.png)

# Python

## Python中的面向对象

### 类与实例

> 参考教程：[1](https://blog.csdn.net/CLHugh/article/details/75000104)

- 面向对象最重要的两个概念是**类**（class）与**实例**（instance）

- **类的创建**：

  ```python
  class Student(object) # 创建Student类
  	pass
  ```

- **类的初始化**：在`__init__`**方法**中创建该类所具有的**属性**（attribute，可以理解为类的“全局变量”）

  ```python
  class Student(object)
  	def __init__(self,name,score)
  		self.name = name # 创建公有变量name
  		self.__score = score # 创建私有变量score
  ```

  - `__init__`方法的第一参数永远是`self`，其指向创建的**类的实例本身**，被称为类的本身实例变量

  - `__init__`方法中的参数指明了**创建实例时需要传入的参数**，但`self`不需要传

  - `__init__`方法将传入的参数绑定给了类/实例本身的参数

  - 私有变量名前加双下划线，如`__score`。公有变量与私有变量的区别在于，前者可以从类的外部访问：`LiHua.name`，而后者不可以（`LiHua.__score`会报错）。针对私有变量，可以在类内部设置**查询变量**与**修改变量**的函数，方便对私有变量的操作：

    ```python
    class Student(object)
    	...
    	def get_score(self):
    		return self.__score
    		
    	def set_score(self,setscore):
    		self.__score = setscore
    ```

    

- **在类中定义函数**：只有一点不同，即第一参数永远是类的本身实例变量`self`，且调用函数时不用传递该参数

- **实例的创建**：

  ```python
  LiHua = Student() # 创建实例LiHua
  ```



## super函数与MRO

> 参考教程：[1](https://zhuanlan.zhihu.com/p/356720970)、[2](https://python3-cookbook.readthedocs.io/zh_CN/latest/c08/p07_calling_method_on_parent_class.html)

### super()

- 直观理解，**super的作用就是执行父类的方法**

- 以单继承为例

  ```
  class A:
      def p(self):
          print('A')
  class B(A):
      def p(self):
          super().p()
  B().p()
  
  >>> A
  ```

  可以看到B().p()其实就是执行的A.p()

- 使用方法：在子类的函数（通常为初始化函数`__init__(self)`）中调用`super().函数名()`（如`super().__init__(self)`）

- 注意，不要用这样的方法代替`父类名.函数名()`，这样会出现重复调用的结果？

### MRO：方法解析顺序列表

- 查询class C的MRO：`C.__mro__`
- MRO是一个所有基类的线性顺序表

## range函数

- 作用：在给定范围内生成数字序列；主要用于for循环
- `range(start, stop, step)`
  - `start`：计数从 start 开始，默认为0
  - `stop`：计数到 stop 结束，**但不包括 stop**
  - `step`：步长，默认为1

- 例如：
  - `range(10)`返回`[0,1,2,3,4,5,6,7,8,9]`
  - `range(0,10,3)`返回`[0,3,6,9]`
  - 用在循环中：`for i = range(3)`意味着index`i`遍历0,1,2

## enumerate函数

- 作用：枚举，将一个可遍历的数据对象（如列表、元组或字符串）组合为一个**索引序列**，同时列出数据和数据下标；主要用于for循环
- `enumerate(sequence, start=i)`
  - `sequence`：一个序列、迭代器或其他支持迭代对象
  - `start`：下标起始位置的值，默认为0

- 例如：

  ```python
  >>> seasons = ['Spring', 'Summer', 'Fall', 'Winter']
  >>> list(enumerate(seasons))
  [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
  >>> list(enumerate(seasons, start=1))       # 下标从 1 开始
  [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
  ```

## if \_\_name\_\_ == '\__main__'

一个python文件通常有两种使用方法，第一是作为脚本直接执行，第二是 import 到其他的 python 脚本中被调用（模块重用）执行。因此` if __name__ == '__main__'`: 的作用就是控制这两种情况执行代码的过程，在 `if __name__ == '__main__'`: 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而 import 到其他脚本中是不会被执行的

## 类中的特殊函数__call\_\_

- 可以通过调用`实例（自变量）`直接调用`__call__`

```python
class Person
	def __call__(self, name)
		print("__call__" + "hello" + name)
		
	def hello(self, name)
		print("hello" + name)
		
person = Person()
person("ZhangSan")  # 直接调用call
person.hello("LiSi")
```

