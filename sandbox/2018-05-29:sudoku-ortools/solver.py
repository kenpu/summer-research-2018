# Research Group Sudoku Solver Solution
import numpy as np
from ortools.constraint_solver import pywrapcp

class Puzzle:
    block_size = 3
    dim = 9

    def __init__(self):
        self.entries = np.empty([self.dim,self.dim], dtype=object)
        self.solver = pywrapcp.Solver("puzzle")
        self.coord = dict()

    def loadf(self, filepath):
        with open(filepath, 'r') as f:
            lines = list(f)
            entries = [x.split() for x in lines]
        assert len(entries) == self.dim
        assert len(entries[0]) == self.dim

        for i in range(len(entries)):
            for j in range(len(entries[i])):
                x = entries[i][j]
                self.entries[i,j] = self.parse_entry(i,j,x)

    def parse_entry(self, i, j, x):
        label = "var%d_%d" % (i,j)
        if x == "_":
            self.coord[label] = (i,j)
            return self.solver.IntVar(1, self.dim, label)
        else:
            return self.solver.IntConst(int(x))

    def make_constraints(self):
        s = self.solver
        e = self.entries
        bs = self.block_size

        # Add Row and Columns Constraints
        for i in range(self.dim):
            s.Add(s.AllDifferent(list(e[i,:])))
            s.Add(s.AllDifferent(list(e[:,i])))

        # Add Block Constraints
        for i in range(self.dim//bs):
            for j in range(self.dim//bs):
                s.Add(s.AllDifferent(list(e[i*bs:(i+1)*bs,j*bs:(j+1)*bs].reshape(bs**2))))

    def solve(self):
        s = self.solver
        e = self.entries
        db = s.Phase(list(e.reshape(self.dim**2)), s.CHOOSE_FIRST_UNBOUND, s.ASSIGN_MIN_VALUE)

        # Starts Solution Search
        s.NewSearch(db)
        # Finds First Solution
        s.NextSolution()
        sol = np.asarray([[np.int64(e[i,j].Value()) for i in range(self.dim)] for j in range(self.dim)], dtype=np.int64)
        # Ends Search
        s.EndSearch()
        # returns first solution found
        return sol

    def __str__(self):
        return str(self.entries)

def main():
    p = Puzzle()
    p.loadf(filepath='../../data/9 by 9.txt')
    p.make_constraints()
    sol = p.solve()
    print(sol)

if __name__ == '__main__':
    main()
