def print_1D_output(filename, output):
    with open(filename, "w") as f:
        f.write("[ ")
        for i, image in enumerate(output):
            for j, row in enumerate(image):
                for k, val in enumerate(row):
                    if (i == len(output) - 1) and (j == len(image) - 1) and (k == len(row) - 1):
                        f.write(f"{int(val)} ]")
                    else:
                        f.write(f"{int(val)}, ")


def print_3D_output(filename, output):
    with open(filename, "w") as f:
        for image in output:
            for row in image:
                f.write("[ ")
                for i, val in enumerate(row):
                    if i == len(row) - 1:
                        f.write(f"{int(val)} ]\n")
                    else:
                        f.write(f"{int(val)}, ")