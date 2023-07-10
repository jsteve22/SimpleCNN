def print_1D_output(filename, output):
    with open(filename, "w") as f:
        f.write("[ ")
        for image in output:
            for row in image:
                for val in row:
                    f.write(f"{int(val)}, ")
        f.write(" ]")


def print_3D_output(filename, output):
    with open(filename, "w") as f:
        for image in output:
            for row in image:
                f.write("[ ")
                for val in row:
                    f.write(f"{int(val)}, ")
                f.write(" ]\n")