import numpy as np

def compare_output(gazelle_file, python_file):
    g_fp = open(gazelle_file, "r")
    p_fp = open(python_file, "r")
    g_nums = []
    p_nums = []
    differences = []
    total_difference = 0
    num_different = 0
    percent_different = 0
    percent_off = 0


    for l in g_fp:
        g_line = l.rstrip()
        g_nums = g_nums + list(map(int, [x.strip(", ") for x in g_line.split(" ")[1:-1]]))
    for l in p_fp:
        p_line = l.rstrip()
        p_nums = p_nums + list(map(int, [x.strip("[] ") for x in p_line.split(",")]))
    
    if len(g_nums) != len(p_nums):
        print("Difference in number of elements")
    else:
        for i in range(len(g_nums)):
            if g_nums[i] != p_nums[i]:
                num_different += 1
                total_difference += abs(g_nums[i] - p_nums[i])
                differences.append(abs(g_nums[i] - p_nums[i]))
        percent_different = (num_different / len(g_nums)) * 100
        print(f"number of different values: {num_different}, number compared: {len(g_nums)}")
        print(f"percent different: {percent_different}")
        print(f"total amount off: {total_difference}")
        print(f"standard deviation: {np.std(differences)}")
        

def accuracy_test():
    files = ["1_conv_output.txt", "2_relu_output.txt", "3_conv_output.txt", "4_relu_output.txt", 
             "5_meanpool_output.txt", "6_conv_output.txt", "7_relu_output.txt", "8_conv_output.txt",
             "9_relu_output.txt", "10_meanpool_output.txt", "11_conv_output.txt", "12_relu_output.txt",
             "13_conv_output.txt", "14_relu_output.txt", "15_conv_output.txt", "16_relu_output.txt"]
    for file in files:
        print(f"comparing {file}")
        compare_output(f"./gazelle_output_files/{file}", f"./output_files/{file}")
        print()

accuracy_test()