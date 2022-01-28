from itertools import combinations
from os import listdir
import re

#---------------------------------CLASS---------------------------------
class Literal: 
    def __init__(self, value):
        self.value = value
        
    def not_literal(self):
        if self.value[0] == '-':
            not_literal = Literal(self.value[1])
        else:
            not_literal = Literal('-' + self.value[0])
        return not_literal
    
    # overloads
    def __repr__(self):
        return self.value
    
    def __eq__(self, other):
        if not isinstance(other, type(self)): 
            return NotImplemented
        return self.value == other.value 
    
    def __hash__(self):
        return hash(self.value)
  
# OR only  
class Clause: # ex: A OR B OR C -> [A,B,C]
    def __init__(self, value):  # ('A', '-B', 'C')
        value = sorted(value, key=lambda lit: lit[0] if len(lit) == 1 else lit[1])
        value = [Literal(lit) for lit in value]
        self.value = value
    
    # return NOT(current_clause)
    def not_clause(self): # not(-A OR B) = not([-A,B]) -> A AND -B = [[A],[-B]] -> 2 seperated clauses 
        clauses = [Clause([lit.not_literal().value]) for lit in self.value]
        return clauses
    
    # remove the first literal occurrence
    def drop_literal(self, literal): 
        res = self.value.copy() # list of Literals
        res.remove(literal) 
        res = [lit.value for lit in res] # list of strings
        return Clause(res) 
        
    # merge current clause with another clause, return new clause
    def merge(self, other):
        # ex: [A,B] merge [B,-C] -> [A,B,-C], not [A,B,B,-C]
        res = list(set(self.value).union(set(other.value))) # list of Literals
        res = [lit.value for lit in res] # list of strings
        return Clause(res) 
    
    # check if clause is TRUE. Ex: A or -A = 1
    def is_true(self):
        for lit_i in self.value:
            for lit_j in self.value:
                if lit_i == lit_j.not_literal():
                    return True
        return False 
    
    def is_empty(self):
        return False if self.value else True
    
    # overloads
    def __repr__(self):
        res = [lit.value for lit in self.value]
        res = ' OR '.join(res)
        return res
    
    def __eq__(self, other):
        if not isinstance(other, type(self)): 
            return NotImplemented
        this = [lit.value for lit in self.value] # list of Literals
        that = [lit.value for lit in other.value]
        return this == that
    
    def __hash__(self):
        this = [lit.value for lit in self.value]
        return hash(tuple(this))

#---------------------------------FUNCTIONS---------------------------------
def read_input(file_name):
    KB = set()
    with open(file_name) as f:
        alpha = Clause(f.readline().rstrip('\n').replace('OR','').split())
        n = int(f.readline().rstrip('\n'))
        for i in range(n):
            clause = Clause(f.readline().rstrip('\n').replace('OR','').split()) # split based on space(s)
            KB.add(clause)
    return alpha, list(KB)

# resolve 2 clauses, if there's nothing to resolve -> return [clause_1, clause_2]
def resolve_pair(c1, c2):
    res = set()
    for literal_1 in c1.value:
        for literal_2 in c2.value:
            if literal_1 == literal_2.not_literal() or literal_1.not_literal() == literal_2:
                tmp1 = c1.drop_literal(literal_1)
                tmp2 = c2.drop_literal(literal_2)
                re = tmp1.merge(tmp2)
                res.add(re)
                
    if not res: # if there's nothing to resolve
        return [c1, c2]
    return list(res) 
    
# return list of resolution steps
def pl_resolution(alpha, KB):
    not_alpha = alpha.not_clause()
    kb = set(KB + not_alpha) # KB ∧ ¬α
    new = set()
    res = []

    while 1:
        # scan kb
        for pair in combinations(kb, 2):
            resolvents = resolve_pair(pair[0], pair[1])
            new = new.union(set(resolvents))
            
        new = set([clause for clause in list(new) if not clause.is_true()])
        new_clauses = new.difference(kb)
        res.append(new_clauses)
        
        # if new contains the empty clause
        for clause in new:
            if clause.is_empty():
                res.append('YES')
                return res
        
        # new ⊆ clauses 
        if new.issubset(kb):
            res.append('NO')
            return res
        
        kb = kb.union(new)

#---------------------------------MAIN---------------------------------
def main():
    inputs = ['./input/' + f for f in listdir('./input') if re.search(r'input\d+.txt', f)]

    for input in inputs:
        alpha, KB = read_input(input)
        reso_res = pl_resolution(alpha, KB)
        id = input[len('./input/input'): input.find('.', 1)]

        with open(f'./output/output{id}.txt', 'w') as f:
            for clause_lst in reso_res[:-1]:
                f.write(str(len(clause_lst)) + '\n')
                for clause in list(clause_lst):
                    content = str(clause) if not clause.is_empty() else '{}'
                    f.write(content + '\n')
            f.write(reso_res[-1])

main()
