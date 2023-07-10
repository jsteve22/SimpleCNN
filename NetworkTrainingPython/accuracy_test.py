def compare_output(gazelle_file, python_file):
    g_fp = open(gazelle_file, "r")
    p_fp = open(python_file, "r")
    g_nums = []
    p_nums = []
    total_difference = 0
    num_different = 0
    percent_different = 0
    percent_off = 0

    if len(g_fp) != len(p_fp):
        print("Difference in file length")
    for l in g_fp:
        g_line = l.readline().rstrip()
        g_nums = g_nums + list(map(int, g_line.split(" ")[1::-1]))
    for l in p_fp:
        p_line = l.readline().rstrip()
        p_nums = p_nums + list(map(int, p_line.split(" ")[1::-1]))

    if len(g_nums) != len(p_nums):
        print("Difference in number of elements")
    else:
        for i in range(len(g_nums)):
            if g_nums[i] != p_nums[i]:
                num_different += 1
                total_difference += abs(g_nums[i] - p_nums[i])
                percent_off += (abs(g_nums[i] - p_nums[i]) / g_nums[i])
        percent_different = num_different / len(g_nums)
        print(f"number of different values: {num_different}")
        print(f"percent different: {percent_different}")
        print(f"total amount off: {total_difference}")
        print(f"percent accuracy: {percent_off}")
        

        