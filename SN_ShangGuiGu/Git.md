视频链接
[尚硅谷Git入门到精通全套教程（涵盖GitHub\Gitee码云\GitLab）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1vy4y1s7k6/?spm_id_from=333.337.search-card.all.click&vd_source=339a4744bd362ae7b381fd9629bfd3a9)

别人比较全的笔记
[(25条消息) Git学习笔记_巨輪的博客-CSDN博客](https://blog.csdn.net/u011863024/article/details/118562748)

# Git
## Git介绍
### Git工作机制
工作区（写代码）--git add-->暂存区（临时存储） --git commit-->本地库（历史版本）

### 代码托管中心
代码托管中心是基于网络服务器的远程代码仓库，一般我们简单称为远程库
1）局域网：GitLab
2）互联网：GitHub、Gitee


## Git命令
### 设置用户签名
```git
git config --global user.name 用户名
git config --global user.email 邮箱

# 可以在c盘用户下面的.gitconfig文件查看
```

### 初始化本地库
```git
# 每次新建一个本地库就要初始化一下
git init
```

### 查看本地库状态
```git
# 红色表示文件还未添加到暂存区
git status
```

### 添加暂存区
```git
git add 文件名

# 从暂存区中删除（只是从暂存区中删除，工作区还存在）
git rm --cached 文件名
```

### 提交本地库
```git
git commit -m "日志信息" 文件名
```

### 查看版本信息
```git
# 查看版本信息
git reflog

# 查看版本详细日志
git log
```

### 版本穿梭
```git
git reset --hard 版本号
```

## Git分支
### 分支的操作
```git
# 创建分支
git branch 分支名

# 查看分支
git branch -v

# 切换分支
git checkout 分支名
git switch 分支名

# 把指定分支合并到当前分支上
git merge 分支名
```

### 合并冲突
```git
# 进入要合并的文件中，修改文件
# 然后添加到暂存区在提交到本地库（这时提交到本地库时不加文件名）
# <<<<到==== 之间是当前分支的代码,====到>>>>是要合并的代码
# 修改并提交完成后，另外那个合并的分支的文件不会发生改变

# 下面是要合并的txt文件
hello, git!
hello, git!
hello, git!
hello, git!
<<<<<<< HEAD
hello, git!hot-fix test
hello, git!master test
=======
hello, git!
hello, git!hot-fix test
>>>>>>> hot-fix
```

## Idea集成Git
### 配置Git忽略文件
与项目的实际功能无关，不参与服务器上部署运行的文件，把它们忽略掉能够屏蔽 IDE 工具之间的差异
创建忽略规则文件 xxxx.ignore（前缀名随便起，建议是git.ignore），这个文件的存放位置原则上在哪里都可以，为了便于让~/.gitconfig 文件引用，**建议**也放在用户家目录下。
模板如下
```git
# Compiled class file
*.class

# Log file
*.log

# BlueJ files
*.ctxt

# Mobile Tools for Java (J2ME)
.mtj.tmp/

# Package Files #
*.jar
*.war
*.nar
*.ear
*.zip
*.tar.gz
*.rar

# virtual machine crash logs, see http://www.java.com/en/download/help/error_hotspot.xml
hs_err_pid*
.classpath
.project
.settings
target
.idea
*.iml
```
在.gitconfig 文件中引用忽略配置文件（此文件在 Windows 的家目录中）
```git
[user]
    name = Layne
    email = Layne@atguigu.com
[core]
# 注意：这里要使用“正斜线（/）”，不要使用“反斜线（\）”
	excludesfile = C:/Users/asus/git.ignore
```
然后再Idea配置Git程序，在菜单栏File->Setting->搜索栏搜Git，配置Git的安装路径（git安装目录下的 git.exe）。

### Idea初始化Git
在菜单栏VCS -> Import into Version Control -> Create Git Repository -> 选择要创建 Git 本地仓库的工程（默认选中的目录就是当前项目的根目录）
添加到暂存区：右键红色的文件（如果是选择根目录就会添加整个项目到根目录，这时候会提示是否强制提交git.ignore中我们要忽略的文件选择cancel，如果没有提示自己找找有没有问题） -> Git -> Add
提交至本地库：选择Git->Commit
（蓝色的文件表示已经追踪，可以不用添加暂存区可以直接提交到本地库）

### 切换版本
查看版本信息：左下角Version Control -> Log
切换版本：右键选择要切换的版本，然后在菜单里点击 Checkout Revision
注意：Idea实现版本切换的功能，并不是使用reset，reset这种方法会直接把HEAD以及MASTER一起拉回那个版本，并且会丢失目标版本之后的内容，虽然可以通过reflog找到后面的版本，但是这种方法比较强烈，可能造成数据丢失。idea使用的是git checkout -b <branch-name> <commit>的方法临时再目标版本新建一个分支切过去看的，再切回来的时候，他会使用git branch -d <branch-name>的方法，再把临时创建的那个仅供会看目的的分支给删除，他的底层是创建一个临时的匿名分支，<commit为版本号>这种方式会让HEAD处于游离状态，恢复HEAD的方式就是用git switch master或者 git checkout master就可以回最新版本。

### 创建分支/切换分支
创建分支：右键项目选择Git -> Repository -> Branches 或者 点击右下角（Git:当前分支）这个图标然后选择点击New Branch
切换分支：跟创建分支步骤相似，点击IDEA的右下角（Git:当前分支）这个图标，选择你想要切换的分支，然后checkout

### 分支合并
合并分支： 点击右下角（Git:当前分支）这个图标然后选择要合并的分支再点击Merge Into Current

# GitHub
![[Pasted image 20230201203708.png]]
![[Pasted image 20230201203521.png]]
## 推送本地库到远程库
```git
# 查看当前所有远程地址别名
git remote -v

# 给远程库创建别名
# 远程地址链接太长了记不住可以起别名
git remote add 别名 远程地址

# 推送本地分支上的内容到远程仓库
# 下面的别名是远程库的别名,分支是本地库的分支
git push 别名 分支

```

## 拉取远程库到本地库
```git
# pull的前提是已经有本地库
git pull 远程库地址别名 远程分支名
```

## 代码克隆  Clone
```git
# 克隆会自动初始化本地库，起别名为origin
git clone 远程地址
```

## SSH免密登录

## Idea集成GitHub
菜单栏File->Setting->搜索栏搜GitHub，添加GitHub账号
由于网络问题，会时常登陆不了，可通过Token登陆：进入github->setting->developer settings->personal access tokens->generate new token(note顺便写，note下面的选项是口令的权限全部打满)
另外一个方法是：![[Pasted image 20230205160116.png]]

将idea的项目直接分享到github（不用创建远程库）：vcs->import into version control->share projetct on github

push本地库到远程库：右键项目->git->commit directory（先提交本地库）
本地库推送到远程库：右键项目->git->repository->push/vcs->git->push（但是这两个都是用https协议)
（ssh）先复制远程库的ssh地址->跟上面一样然后点击跳出来的界面的左上角define remote->然后点同样位置就可以选择使用ssh进行push

pull远程库到本地库：vcs->git->pull

克隆代码到本地：打开idea时点击get from version control

# Gitee码云
## 码云创建远程库


## Idea集成Gitee码云

## 码云连接Gihub 进行代码的复制和迁移


# GitLab
## GitLab服务器的搭建和部署

## Idea集成GitLab