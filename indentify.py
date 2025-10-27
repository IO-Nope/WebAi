
import torch  
from torchvision import models
from torchvision import transforms  
from PIL import Image  
import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # 将模型设置为评估模式（关闭训练专用层）  
model.eval()  
model.to(device)  # 查看模型结构  print(model)  

# 定义图像预处理流程  
preprocess = transforms.Compose([      
	transforms.Resize(256),            # 调整大小      
	transforms.CenterCrop(224),        # 中心裁剪      
	transforms.ToTensor(),             # 转为张量      
	transforms.Normalize(              # 标准化          
		mean=[0.485, 0.456, 0.406],          
		std=[0.229, 0.224, 0.225]      
	)  
])  
err_instance = {}

def load_jpegs(folder_path):
	'''加载文件夹下所有JPEG图像'''
	jpeg_files = ["/".join([folder_path, f]) for f in os.listdir(folder_path) if f.lower().endswith('.jpeg')]
	return jpeg_files

def intestr(str1):
	ans = ""
	for ch in str1:
		if ch in "abcdefghijklmnopqrstuvwxyz":
			ans += ch
	return ans
#utils.Dprint("\n".join(load_jpegs(img_dic_path)))
def fuzzy_match(substring, string):
	"""检查 substring 是否和 string 匹配"""
	Asubstr = intestr(substring.lower())
	Astring = intestr(string.lower())
	if Asubstr in Astring or Astring in Asubstr:
		return True
	return False

def extract_label(img_path):
	'''从图像路径中提取标签'''
	beg = 0 
	end = len(img_path) 
	for i in range(len(img_path)):
		if not beg and img_path[i] == '_':
			beg = i+1
		if img_path[i] == '.':
			end = i
			break
	return img_path[beg:end]

def identify_image(img_path,correct_rate =0.5):
	'''识别图像类别'''
	# 加载类别标签  
	if not os.path.exists('imagenet_classes.txt'):
		with open('imagenet_classes.txt', 'w') as f: 
			f.write(requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json').text)  
	with open('imagenet_classes.txt') as f:      
		classes = [line.strip() for line in f.readlines()]  
	# 图像分类函数  
	img = Image.open(img_path).convert('RGB')     # 训练集有灰度图像，需转换为RGB三通道 
	input_tensor = preprocess(img)      
	input_batch = input_tensor.unsqueeze(0)  # 增加批次维度      
	input_batch = input_batch.to(device)          
	# 执行推理      
	with torch.no_grad():          
		output = model(input_batch)          
	# 计算概率      
	probabilities = torch.nn.functional.softmax(output[0], dim=0)          
	# 获取Top-5结果      
	top5_prob, top5_catid = torch.topk(probabilities, 5)          
	# 显示结果      
	#utils.Dprint(f"\n图像: {img_path.split('/')[-1]} 的识别结果:") 
	 
	types = []
	j = 0
	for i in top5_prob:
		if i.item() >= correct_rate:
			types.append(classes[top5_catid[j]])
		j += 1
	is_match = False
	Ptype = extract_label(img_path)

	for t in types:
		if fuzzy_match(Ptype, t):
			is_match = True
			break
	if is_match:
		utils.Dprint(f"图像: {img_path.split('/')[-1]} 的识别结果: 真实类别为 {Ptype} , 识别正确")
		return True
	else:
		utils.Dprint(f"图像: {img_path.split('/')[-1]} 的识别结果: 真实类别为 {Ptype} , 识别错误，识别为 {classes[top5_catid[0]]}")
		err_instance[img_path] = [Ptype,
							[classes[top5_catid[i]] for i in range(top5_prob.size(0))],
							[ f"{top5_prob[i].item()*100:.2f}%" for i in range(top5_prob.size(0))]
							]
		return False
def exec_train(imgs,correct_rate=0.5):
	'''批量识别图像类别(PR曲线)'''
	su_count = 0
	for img_path in imgs:
		if identify_image(img_path,correct_rate):
			su_count += 1
	return su_count/len(imgs)

def analyze_train(imgs,correct_rate=0.5):
	'''批量识别图像类别(分析和检查错误案例)'''
	su_count = 0
	for img_path in imgs[0:20]:
		if identify_image(img_path,correct_rate):
			su_count += 1
	utils.Dprint(f"\n当前已处理图像数: {len(err_instance)+su_count} , 识别正确数: {su_count} , 识别错误数: {len(err_instance)},正确率为 {su_count/(len(err_instance)+su_count)*100:.2f}% \n")
	with open('error_log.txt', 'w', encoding='utf-8') as f:
		for img_path, info in err_instance.items():
			output = ""
			for i in range(len(info[1])):
				output += f"类别: {info[1][i]} 概率: {info[2][i]} "
			f.write(f"图像: {img_path.split('/')[-1]} 真实类别: {info[0]} 识别结果: {output}\n")
		utils.Dprint("错误日志已保存到 error_log.txt 文件中。")

if __name__ == '__main__':
	img_dic_path = 'D:/Code/Python/WebAi/imgs'   # 填写要批处理的图像的文件夹路径
	imgs = load_jpegs(img_dic_path)
	print(f"共加载到 {len(imgs)} 张 JPEG 图像用于识别。")
	steps = 20
	xlist = np.linspace(0, 1, steps)
	ylist = []
	for rate in xlist:
		acc = exec_train(imgs[:20],correct_rate=rate) # 仅测试前20张图像 因为太多图像会很慢
		ylist.append(acc)
	# 绘制 PR 曲线
	plt.figure()
	plt.plot(xlist, ylist, marker='o', label='PR Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.grid()
	plt.legend(loc='best')
	plt.show()