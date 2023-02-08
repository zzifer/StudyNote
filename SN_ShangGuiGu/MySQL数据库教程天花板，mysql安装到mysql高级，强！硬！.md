[MySQL数据库教程天花板，mysql安装到mysql高级，强！硬！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1iq4y1u7vj/?spm_id_from=333.337.search-card.all.click&vd_source=339a4744bd362ae7b381fd9629bfd3a9)


# 02 MySql 
## 1. MySQL的卸载 p6

## 2. 多版本MySQL的下载、安装、配置 p7

## 3. MySQL的登录 p8

## 4. MySQL的编码设置 p9
### MySQL5.7中
**问题再现：命令行操作sql乱码问题**
```mysql
# 在mysql5.7中默认的字符集为拉丁所以插入带中文的数据时会报错
mysql> INSERT INTO t_stu VALUES(1,'张三','男');
ERROR 1366 (HY000): Incorrect string value: '\xD5\xC5\xC8\xFD' for column 'sname' at row 1
```
解决方法：
步骤1：查看编码命令
```mysql
show variables like 'character_%';  
show variables like 'collation_%';
```

步骤2：修改mysql的数据目录下的my.ini配置文件
```mysql
[mysql]  #大概在63行左右，在其下添加  
...   
default-character-set=utf8  #默认字符集  
​  
[mysqld]  # 大概在76行左右，在其下添加  
...  
character-set-server=utf8  
collation-server=utf8_general_ci
```
> 注意：建议修改配置文件使用notepad++等高级文本编辑器，使用记事本等软件打开修改后可能会导致文件编码修改为“含BOM头”的编码，从而服务重启失败。

步骤3：重启服务
步骤4：查看编码命令
```mysql
show variables like 'character_%';
show variables like 'collation_%';
```

### MySQL8.0中
在MySQL 8.0版本之前，默认字符集为latin1，utf8字符集指向的是utf8mb3。网站开发人员在数据库设计的时候往往会将编码修改为utf8字符集。如果遗忘修改默认的编码，就会出现乱码的问题。从MySQL 8.0开始，数据库的默认编码改为`utf8mb4`，从而避免了上述的乱码问题。

## 5. MySql图形化工具可能出现的连接问题 p10
有些图形界面工具，特别是旧版本的图形界面工具，在连接MySQL8时出现“Authentication plugin 'caching_sha2_password' cannot be loaded”错误。
安装的时候选择第二个选项就不会出错
![[Pasted image 20230208192310.png]]

出现这个原因是MySQL8之前的版本中加密规则是mysql_native_password，而在MySQL8之后，加密规则是caching_sha2_password。解决问题方法有两种，第一种是升级图形界面工具版本，第二种是把MySQL8用户登录密码加密规则还原成mysql_native_password。

第二种解决方案如下，用命令行登录MySQL数据库之后，执行如下命令修改用户密码加密规则并更新用户密码，这里修改用户名为“root@localhost”的用户密码规则为“mysql_native_password”，密码值为“123456”，如图所示。
```mysql
#使用mysql数据库
USE mysql; 

#修改'root'@'localhost'用户的密码规则和密码
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'abc123'; 

#刷新权限
FLUSH PRIVILEGES;
```

## 6. MySql目录结构和源码
### 6.1 主要目录结构
![[Pasted image 20230208194138.png]]

### 6.2 MySQL 源代码获取
首先，你要进入 MySQL下载界面。 这里你不要选择用默认的“Microsoft Windows”，而是要通过下拉栏，找到“Source Code”，在下面的操作系统版本里面， 选择 Windows（Architecture Independent），然后点击下载。
接下来，把下载下来的压缩文件解压，我们就得到了 MySQL 的源代码。
MySQL 是用 C++ 开发而成的，我简单介绍一下源代码的组成。
mysql-8.0.22 目录下的各个子目录，包含了 MySQL 各部分组件的源代码：
![[Pasted image 20230208194226.png]]


# 03 SELECT 12-17
## 1. SQL分类
SQL语言在功能上主要分为如下3大类：
-   **DDL（Data Definition Languages、数据定义语言）**，这些语句定义了不同的数据库、表、视图、索引等数据库对象，还可以用来创建、删除、修改数据库和数据表的结构。
    -   主要的语句关键字包括`CREATE`、`DROP`、`ALTER`等。
-   **DML（Data Manipulation Language、数据操作语言）**，用于添加、删除、更新和查询数据库记录，并检查数据完整性。
    -   主要的语句关键字包括`INSERT`、`DELETE`、`UPDATE`、`SELECT`等。
    -   **SELECT是SQL语言的基础，最为重要。**
-   **DCL（Data Control Language、数据控制语言）**，用于定义数据库、表、字段、用户的访问权限和安全级别。
    -   主要的语句关键字包括`GRANT`、`REVOKE`、`COMMIT`、`ROLLBACK`、`SAVEPOINT`等。

>因为查询语句使用的非常的频繁，所以很多人把查询语句单拎出来一类：DQL（数据查询语言）。
>还有单独将`COMMIT`、`ROLLBACK` 取出来称为TCL （Transaction Control Language，事务控制语言）。

## 2. 使用规范和数据导入
### 2.1 基本规则
* SQL 可以写在一行或者多行。为了提高可读性，各子句分行写，必要时使用缩进 
* 每条命令以 ; 或 \g 或 \G 结束 
* 关键字不能被缩写也不能分行 
* 关于标点符号 
  * 必须保证所有的()、单引号、双引号是成对结束的 
  * 必须使用英文状态下的半角输入方式 
  * 字符串型和日期时间类型的数据可以使用单引号（' '）表示 
  * 列的别名，尽量使用双引号（" "），而且不建议省略as
  
### 2.2 SQL大小写规范（建议遵守）
* MySQL 在 Windows 环境下是大小写不敏感的 
* MySQL 在 Linux 环境下是大小写敏感的 
  * 数据库名、表名、表的别名、变量名是严格区分大小写的 
  * 关键字、函数名、列名(或字段名)、列的别名(字段的别名) 是忽略大小写的。 
* 推荐采用统一的书写规范： 
  * 数据库名、表名、表别名、字段名、字段别名等都小写 
  * SQL 关键字、函数名、绑定变量等都大写

### 2.3 注释
```mysql
单行注释：#注释文字(MySQL特有的方式)
单行注释：-- 注释文字(--后面必须包含一个空格。)
多行注释：/* 注释文字 */
```

### 2.4 命名规则
* 数据库、表名不得超过30个字符，变量名限制为29个 
* 必须只能包含 A–Z, a–z, 0–9, _共63个字符 
* 数据库名、表名、字段名等对象名中间不要包含空格 同一个MySQL软件中，数据库不能同名；同一个库中，表不能重名；
* 同一个表中，字段不能重名 必须保证你的字段没有和保留字、数据库系统或常用方法冲突。如果坚持使用，请在SQL语句中使 用`（着重号）引起来 
* 保持字段名和类型的一致性，在命名字段并为其指定数据类型的时候一定要保证一致性。假如数据 类型在一个表里是整数，那在另一个表里可就别变成字符型了

### 2.5 导入现有表
```mysql
source 路径
```

## 3. 基本的SELECT语句
### 3.0 SELECT...
```mysql
SELECT 1; #没有任何子句  
SELECT 9/2; #没有任何子句
```

### 3.1 SELECT ... FROM
```mysql
SELECT   标识选择哪些列
FROM     标识从哪个表中选择
```
-   选择全部列：
```mysql
SELECT *
FROM   departments;
```
>一般情况下，除非需要使用表中所有的字段数据，最好不要使用通配符‘*’。使用通配符虽然可以节省输入查询语句的时间，但是获取不需要的列数据通常会降低查询和所使用的应用程序的效率。通配符的优势是，当不知道所需要的列的名称时，可以通过它获取它们。
>在生产环境下，不推荐你直接使用`SELECT *`进行查询。

-   选择特定的列：
```mysql
SELECT department_id, location_id
FROM   departments;
```
>MySQL中的SQL语句是不区分大小写的，因此SELECT和select的作用是相同的，但是，许多开发人员习惯将关键字大写、数据列和表名小写，读者也应该养成一个良好的编程习惯，这样写出来的代码更容易阅读和维护。

### 3.2 列的别名
- 在列名和别名之间加入关键字AS（AS可以省略）或者使用""
```mysql
SELECT last_name AS name, commission_pct comm, usrid "name"
FROM employees;
```

### 3.3 去除重复行
>默认情况下，查询会返回全部行，包括重复行
>在SELECT语句中使用关键字DISTINCT去除重复行
```mysql
# 去重DISTINCT
SELECT DISTINCT department_id
FROM   employees;
```
这里有两点需要注意：
1.  DISTINCT 需要放到所有列名的前面，如果写成`SELECT salary, DISTINCT department_id FROM employees`会报错。
2.  DISTINCT 其实是对后面所有列名的组合进行去重，你能看到最后的结果是 74 条，因为这 74 个部门id不同，都有 salary 这个属性值。如果你想要看都有哪些不同的部门（department_id），只需要写`DISTINCT department_id`即可，后面不需要再加其他的列名了。

### 3.4 空值参与运算
-   所有运算符或列值遇到null值，运算的结果都为null
```mysql
SELECT employee_id,salary,commission_pct,
12 * salary * (1 + commission_pct) "annual_sal"
FROM employees;
```
>这里你一定要注意，在 MySQL 里面， 空值不等于空字符串。一个空字符串的长度是 0，而一个空值的长度是空。而且，在 MySQL 里面，空值是占用空间的。

### 3.5 着重号
我们需要保证表中的字段、表名等没有和保留字、数据库系统或常用方法冲突。如果真的相同，请在SQL语句中使用一对``（着重号）引起来
```mysql
mysql
SELECT * FROM `order`
# 查询常数
```

### 3.6 查询常数
SELECT 查询还可以对常数进行查询。对的，就是在 SELECT 查询结果中增加一列固定的常数列。这列的取值是我们指定的，而不是从数据表中动态取出的。
你可能会问为什么我们还要对常数进行查询呢？
SQL 中的 SELECT 语法的确提供了这个功能，一般来说我们只从一个表中查询数据，通常不需要增加一个固定的常数列，但如果我们想整合不同的数据源，用常数列作为这个表的标记，就需要查询常数。
比如说，我们想对 employees 数据表中的员工姓名进行查询，同时增加一列字段`corporation`，这个字段固定值为“尚硅谷”，可以这样写：
```mysql
SELECT '尚硅谷' as corporation, last_name FROM employees;
```

## 4. 显示表结构
```mysql
# 显示表结构
DESCRIBE employees;
或
DESC employees;
```

## 5. where过滤数据
使用WHERE 子句，将不满足条件的行过滤掉。WHERE子句紧随 FROM子句。
```mysql
SELECT employee_id, last_name, job_id, department_id
FROM employees
WHERE department_id = 90;
```

# 04 运算符
### 算术运算符
```mysql
mysql> SELECT 100, 100 + 0, 100 - 0, 100 + 50, 100 + 50 -30, 100 + 35.5, 100 - 35.5 FROM dual;
+-----+---------+---------+----------+--------------+------------+------------+
| 100 | 100 + 0 | 100 - 0 | 100 + 50 | 100 + 50 -30 | 100 + 35.5 | 100 - 35.5 |
+-----+---------+---------+----------+--------------+------------+------------+
| 100 |     100 |     100 |      150 |          120 |      135.5 |       64.5 |
+-----+---------+---------+----------+--------------+------------+------------+
1 row in set (0.00 sec)
```
> - 一个整数类型的值对整数进行加法和减法操作，结果还是一个整数；
> - 一个整数类型的值对浮点数进行加法和减法操作，结果是一个浮点数；
> - 加法和减法的优先级相同，进行先加后减操作与进行先减后加操作的结果是一样的；
> - 在Java中，+的左右两边如果有字符串，那么表示字符串的拼接。但是在MySQL中+只表示数值相加。如果遇到非数值类型，先尝试转成数值，如果转失败，就按0计算。（补充：MySQL中字符串拼接要使用字符串函数CONCAT()实现）


## 比较运算符
1) 等号运算符 =
	比较的结果为真则返回1，比较的结果 为假则返回0，其他情况则返回NULL
	如果等号两边的值、字符串或表达式中有一个为NULL，则比较结果为NULL
	如果等号两边的值、字符串或表达式都为字符串，则MySQL会按照字符串进行比较，其比较的 是每个字符串中字符的ANSI编码是否相等。 
	如果等号两边的值都是整数，则MySQL会按照整数来比较两个值的大小。 
	如果等号两边的值一个是整数，另一个是字符串，则MySQL会将字符串转化为数字进行比较。 
	如果等号两边的值、字符串或表达式中有一个为NULL，则比较结果为NULL。
2) 安全等于 <=>
	安全等于运算符（<=>）与等于运算符（=）的作用是相似的，`唯一的区别`是‘<=>’可以用来对NULL进行判断。在两个操作数均为NULL时，其返回值为1，而不为NULL；当一个操作数为NULL时，其返回值为0，而不为NULL
3) 不等于 <>和!=
4) 非符号类型运算符
![[Pasted image 20230206202209.png]]
5）空运算符
	空运算符 (IS NULL 或者 ISNULL) 判断一个值是否为NULL，如果为NULL则返回1，否则返回0。
```mysql
mysql> SELECT NULL IS NULL, ISNULL(NULL)
```
6）最小值运算符
	语法格式为：LEAST(值1，值2，...，值n)。其中，“值n”表示参数列表中有n个值。在有 两个或多个参数的情况下，返回最小值。
	由结果可以看到，当参数是整数或者浮点数时，LEAST将返回其中最小的值；当参数为字符串时，返回字 母表中顺序最靠前的字符；当比较值列表中有NULL时，不能判断大小，返回值为NULL
```mysql
mysql> SELECT LEAST (1,0,2), LEAST('b','a','c'), LEAST(1,NULL,2);
+---------------+--------------------+-----------------+
| LEAST (1,0,2) | LEAST('b','a','c') | LEAST(1,NULL,2) |
+---------------+--------------------+-----------------+
|       0       |          a         |        NULL     |
+---------------+--------------------+-----------------+
```
7）最大值运算符
	语法格式为：GREATEST(值1，值2，...，值n)。其中，n表示参数列表中有n个值。当有 两个或多个参数时，返回值为最大值。假如任意一个自变量为NULL，则GREATEST()的返回值为NULL
	由结果可以看到，当参数中是整数或者浮点数时，GREATEST将返回其中最大的值；当参数为字符串时， 返回字母表中顺序最靠后的字符；当比较值列表中有NULL时，不能判断大小，返回值为NULL。
```mysql
mysql> SELECT GREATEST(1,0,2), GREATEST('b','a','c'), GREATEST(1,NULL,2);
+-----------------+-----------------------+--------------------+
| GREATEST(1,0,2) | GREATEST('b','a','c') | GREATEST(1,NULL,2) |
+-----------------+-----------------------+--------------------+
|         2       |             c         |         NULL       |
+-----------------+-----------------------+--------------------+
```
8) BETWEEN AND运算符
	BETWEEN运算符使用的格式通常为SELECT D FROM TABLE WHERE C BETWEEN A AND B，此时，当C大于或等于A，并且C小于或等于B时，结果为1，否则结果为0。
9) IN运算符
	IN运算符用于判断给定的值是否是IN列表中的一个值，如果是则返回1，否则返回0。如果给 定的值为NULL，或者IN列表中存在NULL，则结果为NULL
```mysql
mysql> SELECT 'a' IN ('a','b','c'), 1 IN (2,3), NULL IN ('a','b'), 'a' IN ('a', NULL);
+----------------------+------------+-------------------+--------------------+
| 'a' IN ('a','b','c') | 1 IN (2,3) | NULL IN ('a','b') | 'a' IN ('a', NULL) |
+----------------------+------------+-------------------+--------------------+
|            1         |      0     |         NULL      |          1         |
+----------------------+------------+-------------------+--------------------+
```
9) NOT IN运算符
	NOT IN运算符用于判断给定的值是否不是IN列表中的一个值，如果不是IN列表中的一 个值，则返回1，否则返回0。
10) LIKE运算符
	LIKE运算符主要用来匹配字符串，通常用于模糊匹配，如果满足条件则返回1，否则返回 0。如果给定的值或者匹配条件为NULL，则返回结果为NULL。
```mysql
“%”：匹配0个或多个字符。
“_”：只能匹配一个字符。
```
11) ESCAPE
	 回避特殊符号的：使用转义符。例如：将[%]转为[$%]、[]转[$]，然后再加上[ESCAPE‘$’]即可

## 逻辑运算符


## 位运算符


## 运算符的优先级


## 使用正则表达式查询


# 05 排序与分页 22-24

## 1. 排序数据
如果没有使用排序操作，默认情况下查询返回的数据是按照添加数据数据的顺序显示

### 1.1 排序规则
-   使用 ORDER BY 子句排序
    -   **ASC（ascend）: 升序**
    -   **DESC（descend）:降序**
-   ORDER BY 子句在SELECT语句的结尾。

### 1.2 单列排序
```mysql
# 如果在ORDER BY后面没有显示指定排序的方式则默认按照升序排列
# 并且可以使用列的别名进行排序，但是类的别名只能在order by中使用不能再where中使用
# 因为sql执行的顺序是from where select order by，where使用别名时还没有创建别名
# where需要声明在from后面order by前面
SELECT   last_name, job_id, department_id, hire_date  
FROM     employees  
ORDER BY hire_date ;
```

### 1.3 多列排序
```
SELECT last_name, department_id, salary  
FROM   employees  
ORDER BY department_id, salary DESC;
```
-   可以使用不在SELECT列表中的列排序。
-   在对多列进行排序的时候，首先排序的第一列必须有相同的列值，才会对第二列进行排序。如果第一列数据中所有值都是唯一的，将不再对第二列进行排序。

## 2. 分页

### 2.1 背景
- 背景1：查询返回的记录太多了，查看起来很不方便，怎么样能够实现分页查询呢？
- 背景2：表里有 4 条数据，我们只想要显示第 2、3 条数据怎么办呢？

### 2.2 实现规则
-   分页原理
    所谓分页显示，就是将数据库中的结果集，一段一段显示出来需要的条件。
-   **MySQL中使用 LIMIT 实现分页**
-   格式：
```mysql
LIMIT [位置偏移量,] 行数
```
    第一个“位置偏移量”参数指示MySQL从哪一行开始显示，是一个可选参数，如果不指定“位置偏移   量”，将会从表中的第一条记录开始；第二个参数“行数”指示返回的记录条数。
-   举例
```
--前10条记录：  
SELECT * FROM 表名 LIMIT 0,10;  
或者  
SELECT * FROM 表名 LIMIT 10;  
​  
--第11至20条记录：  
SELECT * FROM 表名 LIMIT 10,10;  
​  
--第21至30条记录：   
SELECT * FROM 表名 LIMIT 20,10;
```

> MySQL 8.0中可以使用“LIMIT 3 OFFSET 4”，意思是获取从第5条记录开始后面的3条记录，和“LIMIT 4,3;”返回的结果相同。
-   分页显式公式：（当前页数-1）* 每页条数，每页条数
```
SELECT * FROM table   
LIMIT(PageNo - 1)*PageSize,PageSize;
```
-   **注意：LIMIT 子句必须放在整个SELECT语句的最后！**

### 2.3 拓展
在不同的 DBMS 中使用的关键字可能不同。在 MySQL、PostgreSQL、MariaDB 和 SQLite 中使用 LIMIT 关键字，而且需要放到 SELECT 语句的最后面。

# 06 多表查询 25
多表查询，也称为关联查询，指两个或更多个表一起完成查询操作。

前提条件：这些一起查询的表之间是有关系的（一对一、一对多），它们之间一定是有关联字段，这个关联字段可能建立了外键，也可能没有建立外键。比如：员工表和部门表，这两个表依靠“部门编号”进行关联。

## 1. 一个案例引发的多表连接
### 1.1 案例说明
![[Pasted image 20230208160458.png]]
![[Pasted image 20230208160525.png]]
```mysql
#案例：查询员工的姓名及其部门名称
SELECT last_name, department_name
FROM employees, departments;

查询结果：
+-----------+----------------------+  
| last_name | department_name      |  
+-----------+----------------------+  
| King      | Administration       |  
| King      | Marketing            |  
| King      | Purchasing           |  
| King      | Human Resources      |  
| King      | Shipping             |  
| King      | IT                   |  
| King      | Public Relations     |  
| King      | Sales                |  
| King      | Executive            |  
| King      | Finance              |  
| King      | Accounting           |  
| King      | Treasury             |  
...  
| Gietz     | IT Support           |  
| Gietz     | NOC                  |  
| Gietz     | IT Helpdesk          |  
| Gietz     | Government Sales     |  
| Gietz     | Retail Sales         |  
| Gietz     | Recruiting           |  
| Gietz     | Payroll              |  
+-----------+----------------------+  
2889 rows in set (0.01 sec)

# 分析错误情况：
SELECT COUNT(employee_id) FROM employees;  
#输出107行  
SELECT COUNT(department_id)FROM departments;  
#输出27行  

我们把上述多表查询中出现的问题称为：笛卡尔积的错误
```

### 1.2 笛卡尔积（或交叉连接）的理解
SQL92中，笛卡尔积也称为`交叉连接`，英文是 `CROSS JOIN`。在 SQL99 中也是使用 CROSS JOIN表示交叉连接。它的作用就是可以把任意表进行连接，即使这两张表不相关。在MySQL中如下情况会出现笛卡尔积：
```mysql
#查询员工姓名和所在部门名称
SELECT last_name,department_name FROM employees,departments;
SELECT last_name,department_name FROM employees CROSS JOIN departments;
SELECT last_name,department_name FROM employees INNER JOIN departments;
SELECT last_name,department_name FROM employees JOIN departments;
```

### 1.3 案例分析与问题解决
-   **笛卡尔积的错误会在下面条件下产生**：
    -   省略多个表的连接条件（或关联条件）
    -   连接条件（或关联条件）无效
    -   所有表中的所有行互相连接
-   为了避免笛卡尔积， 可以**在 WHERE 加入有效的连接条件。**
-   加入连接条件后，查询语法：
```
	SELECT  table1.column, table2.column  
    FROM    table1, table2  
    WHERE   table1.column1 = table2.column2;  #连接条件
```
-   在表中有相同列时，在列名之前加上表名前缀
-   可以给表起别名，当使用别名时后续使用到表名时必须也使用表的别名

# MySql架构篇



# 索引及调优篇



# 事务篇 161-
## 13 事务的基础知识 161-
### 基本概念
**事务：一组逻辑操作单元，使数据从一种状态变换到另一种状态。
**事务处理的原则：保证所有事务都作为 `一个工作单元` 来执行，即使出现了故障，都不能改变这种执行方 式。当在一个事务中执行多个操作时，要么所有的事务都被提交( `commit` )，那么这些修改就 `永久` 地保 `存下来`；要么数据库管理系统将 `放弃` 所作的所有 `修改` ，整个事务回滚( rollback )到最初状态。

### 事务的ACID特性
**原子性（atomicity）：**
原子性是指事务是一个不可分割的工作单位，要么全部提交，要么全部失败回滚。即要么转账成功，要么转账失败，是不存在中间的状态。如果无法保证原子性会怎么样？就会出现数据不一致的情形，A账户减去100元，而B账户增加100元操作失败，系统将无故丢失100元。

**一致性（consistency）：**
（国内很多网站上对一致性的阐述有误，具体可以参考 Wikipedia 对Consistency的阐述）
根据定义，一致性是指事务执行前后，数据从一个 `合法性状态` 变换到另外一个 `合法性状态` 。这种状态是 `语义上` 的而不是语法上的，跟具体的业务有关。
那什么是合法的数据状态呢？满足 `预定的约束` 的状态就叫做合法的状态。通俗一点，这状态是由你自己来定义的（比如满足现实世界中的约束）。满足这个状态，数据就是一致的，不满足这个状态，数据就 是不一致的！如果事务中的某个操作失败了，系统就会自动撤销当前正在执行的事务，返回到事务操作 之前的状态。
**举例1：**A账户有200元，转账300元出去，此时A账户余额为-100元。你自然就发现此时数据是不一致的，为什么呢？因为你定义了一个状态，余额这列必须>=0。
**举例2：**A账户有200元，转账50元给B账户，A账户的钱扣了，但是B账户因为各种意外，余额并没有增加。你也知道此时的数据是不一致的，为什么呢？因为你定义了一个状态，要求A+B的总余额必须不变。
**举例3：**在数据表中我们将`姓名`字段设置为`唯一性约束`，这时当事务进行提交或者事务发生回滚的时候，如果数据表的姓名不唯一，就破坏了事物的一致性要求。

**隔离型（isolation）：**
事务的隔离性是指一个事务的执行`不能被其他事务干扰`，即一个事务内部的操作及使用的数据对`并发`的其他事务是隔离的，并发执行的各个事务之间不能相互干扰。

如果无法保证隔离性会怎么样？假设A账户有200元，B账户0元。A账户往B账户转账两次，每次金额为50 元，分别在两个事务中执行。如果无法保证隔离性，会出现下面的情形：
```mysql
UPDATE accounts SET money = money - 50 WHERE NAME = 'AA';
UPDATE accounts SET money = money + 50 WHERE NAME = 'BB';
```

**持久性（durability）：**
持久性是指一个事务一旦被提交，它对数据库中数据的改变就是 永久性的 ，接下来的其他操作和数据库 故障不应该对其有任何影响。

持久性是通过 `事务日志` 来保证的。日志包括了 `重做日志` 和 `回滚日志` 。当我们通过事务对数据进行修改 的时候，首先会将数据库的变化信息记录到重做日志中，然后再对数据库中对应的行进行修改。这样做 的好处是，即使数据库系统崩溃，数据库重启后也能找到没有更新到数据库系统中的重做日志，重新执 行，从而使事务具有持久性。
> 总结
>
> ACID是事务的四大特征，在这四个特性中，原子性是基础，隔离性是手段，一致性是约束条件， 而持久性是我们的目的。
>
> 数据库事务，其实就是数据库设计者为了方便起见，把需要保证`原子性`、`隔离性`、`一致性`和`持久性`的一个或多个数据库操作称为一个事务。


### 事务的状态
我们现在知道 `事务` 是一个抽象的概念，它其实对应着一个或多个数据库操作，MySQL根据这些操作所执 行的不同阶段把 `事务` 大致划分成几个状态：
* **活动的（active）**
  事务对应的数据库操作正在执行过程中时，我们就说该事务处在 `活动的` 状态。
* **部分提交的（partially committed）**
  当事务中的最后一个操作执行完成，但由于操作都在内存中执行，所造成的影响并 `没有刷新到磁盘` 时，我们就说该事务处在 `部分提交的` 状态。
* **失败的（failed）**
  当事务处在 `活动的` 或者 部分提交的 状态时，可能遇到了某些错误（数据库自身的错误、操作系统 错误或者直接断电等）而无法继续执行，或者人为的停止当前事务的执行，我们就说该事务处在 失 败的 状态。
* **中止的（aborted）**
  如果事务执行了一部分而变为 `失败的` 状态，那么就需要把已经修改的事务中的操作还原到事务执 行前的状态。换句话说，就是要撤销失败事务对当前数据库造成的影响。我们把这个撤销的过程称之为 `回滚` 。当 `回滚` 操作执行完毕时，也就是数据库恢复到了执行事务之前的状态，我们就说该事 务处在了 `中止的` 状态。
  举例：
  ```mysql
  UPDATE accounts SET money = money - 50 WHERE NAME = 'AA';
  
  UPDATE accounts SET money = money + 50 WHERE NAME = 'BB';
  ```
* **提交的（committed）**
  当一个处在 `部分提交的` 状态的事务将修改过的数据都 `同步到磁盘` 上之后，我们就可以说该事务处在了 `提交的` 状态。

  一个基本的状态转换图如下所示：
![[Pasted image 20230206202226.png]]
  图中可见，只有当事物处于`提交的`或者`中止的`状态时，一个事务的生命周期才算是结束了。对于已经提交的事务来说，该事务对数据库所做的修改将永久生效，对于处于中止状态的事物，该事务对数据库所做的所有修改都会被回滚到没执行该事物之前的状态。




 




# 日志与备份篇
