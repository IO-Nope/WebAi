def simple_sort(arr: list) -> list:
    list_sorted = sorted(arr)
    return list_sorted
#? 我当然不会在python里手写排序算法

if __name__ == "__main__":
    nums = []
    print("请输入10个整数，用空格分隔：")
    input_str = input()
    nums = list(map(int, input_str.split()))
    if len(nums) != 10:
        print("请输入恰好10个整数！")
    else:
        sorted_nums = simple_sort(nums)
        print("排序后的结果是：", sorted_nums)