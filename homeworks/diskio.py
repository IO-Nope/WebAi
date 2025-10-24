input_str = ""
with open("test.txt", "w") as f:
    while input_str != '#':
        input_str = input("请输入要写入文件的内容：")
        f.write(input_str)
    else:
        print("输入结束，文件已写入。")