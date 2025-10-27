# <center>WebAi
<center> A lesson for AI+ exercise </center>

---

## Lab 1: 初始环境配置


- [x] miniconda_313_x86_64
- [x] 虚拟环境ai_env
- [x] matplotlib,numpy
- [x] Jupyter Notebook


### 作业:
> 作业都放在homeworks/目录下
- [x] 波那契数列/Fibonacci sequence
> fibnums.py
- [x] 对十个数排序/sort
> simplesort.py
- [x] 磁盘读写练习/diskIO
> diskio.py


---

## Lab 2: 预训练模型推理
> ### Issues:
> - ppt中 torch.cuda.is_available() else ‘cpu’,img_path = 'test_images/xxx.JPEG' 处半角全角错了 
> - 两段jupyter notebook代码中变量名由img_path改为image_path
> - 不知为何的动态变量无法检测
### todo:
> identify.py
- [x] 将训练过程函数化
- [x] 读取同一目录下的JPEG文件
- [x] 检测所有读取的文件
- [x] 根据校对检测是否正确，计数正确值,将失败的案例单独输出到一文件
- [x] 添加准确值检测
- [x] 利用PLT绘制PR曲线
- [ ] 添加时间检测
> 时间检测不加了，用的cpu训练效率依托四

### 作业:
- [x] 写一份ResNet-18性能技术报告。尝试绘制PR曲线，分析失败案例。
```
> 报告就不写了，identify.py集成了绘制PR曲线的功能
> 但对于失败案例，虽然对性能对比没什么影响，但刚开始大部分失败是检测词和标注的不一致导致的，已经优化
> 在有相似特征的图片上更容易判断失误，尤其是高回归值时
```
- [x] 将ResNet-18切换至ResNet-50，然后尝试比较二者性能优劣
```
> 更改model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
> 然后删除C:\Users\<Username>\.cache\torch\hub\checkpoints\下的模型权重文件
> 显然的，后者优于前者，我用小样本得出这个结论
```
- [ ] 在ResNet-18网络内加入注意力机制。然后尝试比较三者性能优劣