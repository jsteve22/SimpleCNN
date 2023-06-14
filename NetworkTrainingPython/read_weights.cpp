#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

using namespace std;

int read_weights(std::string file) {
    ifstream fp;
    string line;

    fp.open(file);
    if (fp.is_open()) {
        getline(fp, line);

        int i = 0;
        string dim = "";
        vector<int> dims;
        while (line[i] != '\0') {
            if (line[i] != ' ') {
                dim += line[i];
            }
            else {
            // cout << dim << " ";;
                dims.push_back(stoi(dim));
                dim = "";
            }
            i++;
        }
        dims.push_back(stoi(dim));

        for (int i = 0; i < dims.size(); i++) {
            cout << dims[i];
        }

        if (dims.size() == 1)
    }
    return 0;
}
int main(void) {
    read_weights("conv2d.kernel.txt");
    return 0;
}