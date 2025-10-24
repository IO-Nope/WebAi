def fibs(num:int)->int:
    if num <=1:
        return num
    else:
        return fibs(num-1) + fibs(num-2)
    
if __name__ == "__main__":
    n = input("请输入一个整数n，计算第n个斐波那契数：")
    n = int(n)
    print(f"{n}对应的斐波那契数是：{fibs(n)}")
    