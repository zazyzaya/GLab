#include <string>
#include <fstream>

using namespace std;

// Uses CSR format with edge weight encoded into neighbor id's 
class Graph {
    public: 
        int n_nodes;
        int n_edges;

    private:
        int* offsets;
        int* edges;

        void load_graph(string fname) {
            ifstream f;
            f.open(fname);

            int edge_ct = 0;
            string line;
            while( getline(f, line) ){
                
            }
        }

    public: 
        Graph(string fname) {

        }
}