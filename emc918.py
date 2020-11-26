import sys
import math
import os
import time
import random
import copy

# --------------------
# OBJ : operations
# DESC: Dictionary object in which the keys are a string expression and the item is a lambda function.
# --------------------
operations = {
	"const": lambda p: p[0],
	"add"  : lambda p: p[0].evaluate() + p[1].evaluate(),
	"sub"  : lambda p: p[0].evaluate() - p[1].evaluate(),
	"mul"  : lambda p: p[0].evaluate() * p[1].evaluate(),
	"div"  : lambda p: (lambda x, y: x.evaluate()/y if y != 0 else 0)(p[0],p[1].evaluate()),
	"pow"  : lambda p: (lambda x,y: (lambda n,m: 0 if (n < 0 and not float(m).is_integer()) or (n == 0 and m < 0) else float(n) ** float(m))(x.evaluate(), y) if y != 0 else 1)(p[0], p[1].evaluate()),
	"sqrt" : lambda p: (lambda x: math.sqrt(x) if x >= 0 else 0)(p[0].evaluate()),
	"log"  : lambda p: (lambda x: math.log(x, 2) if x > 0 else 0)(p[0].evaluate()),
	"exp"  : lambda p: (lambda x: x if x != float("inf") else 0)(math.exp(p[0].evaluate())),
	"max"  : lambda p: max(p[0].evaluate(), p[1].evaluate()),
	"ifleq": lambda p: p[2].evaluate() if p[0].evaluate() <= p[1].evaluate() else p[3].evaluate(),
	"data" : lambda p, inp=[], n=0: inp[abs(math.floor(p[0].evaluate()))%n] if n != 0 else 0,
	"diff" : lambda p, inp=[], n=0: inp[abs(math.floor(p[0].evaluate()))%n] - inp[abs(math.floor(p[1].evaluate()))%n] if n != 0 else 0,
	"avg"  : lambda p, inp=[], n=0: (lambda x, y, inp: (lambda i, j, inp: sum(inp[i:j])/(j-i))(min(x,y),max(x,y),inp) if x != y else 0)
				(math.floor(abs(p[0].evaluate()))%n, math.floor(abs(p[1].evaluate()))%n,inp) if n != 0 else 0
}

op_branches = {"add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2, "sqrt": 1, "log": 1, "exp": 1, "max": 2, "ifleq": 4, "data": 1, "diff": 2, "avg": 2}

# --------------------
# OBJ : Node
# DESC: Class representing a tree node. Contains operation, children (vals), size of subtree and fitness.
# --------------------
class Node():
	def __init__(self, _op, _vals, _size=0, _depth=0, _fitness=0):
		self.op   = _op
		self.vals = _vals
		self.size = _size
		self.depth = _depth
		self.fitness = _fitness

	def evaluate(self):
		try:
			ans = operations[self.op](self.vals)
			if ans == float('inf') or math.isnan(ans): return 0
			else: return ans
		except OverflowError:
			return 0

	def display(self):
		print(out_tree(self))

# --------------------
# FUNC  : main
# RETURN: None
# DESC  : Parses command line args and calls function based on questions parameter.
# --------------------
def main(args):

	n = 0
	inp = []

	try:
		question = int(args[args.index("-question")+1])
		assert question <= 3 and question >= 1
	except:
		print("Question not specified. Include the flag '-question' followed by a number in {1, 2, 3}.")
		raise

	try:
		n = int(args[args.index("-n")+1])
	except:
		print("Dimension size not specified. Include the flag '-n' followed by the dimensions of the input vector.")
		raise

	if question == 1 or question == 2:
		try:
			exp = args[args.index("-expr")+1]
		except:
			print("Expression not specified. Include the flag '-expr' followed by an expression to evaluate.")
			raise

	if question == 2 or question == 3:

		try:
			m = int(args[args.index("-m")+1])
		except:
			print("Training data dimension not specified. Include the flag '-m' followed by the dimensions of the training data.")
			raise

		try:
			data = args[args.index("-data")+1]
			assert os.path.isfile(data)
			x, y = read_file(data)
			assert len(x) == m and len(y) == m
			assert all(len(i) == n for i in x)
		except:
			print("Data file error. Include the flag '-data' followed a filename containing data of dimension m by n+1.")
			raise

	if question == 1:

		try:
			x   = args[args.index("-x")+1].split(' ')
			inp = [float(i) for i in x]
			assert len(inp) == n
		except:
			print("Input vector error. Include the flag '-x' followed by a space seperated list of floats, of the same dimension as n.")
			raise

		print(create_tree(exp, inp, n).evaluate())

	elif question == 2:
		tree = parse_exp(exp)
		print(fitness(tree,x,y,m,n))

	elif question == 3:

		try:
			pop_size = int(args[args.index("-lambda")+1])
			assert pop_size > 0
		except:
			print("Population size not specified. Include the flag '-lambda' followed by the population size.")
			raise

		try:
			time_budget = int(args[args.index("-time_budget")+1])
			assert time_budget > 0
		except:
			print("Time budget not specified. Include the flag '-time_budget' followed by the max time in seconds.")
			raise

		a = generate_exp(pop_size, n, m, x, y, time_budget)
		print(output_exp(a))
		# print(a.fitness)

# --------------------
# FUNC  : create_tree
# RETURN: Node Object
# DESC  : Generates tree based on input string.
# --------------------
def create_tree(exp, inp, n):
	refresh_lambda(inp, n)
	return parse_exp(exp)

# --------------------
# FUNC  : parse_exp
# RETURN: Node Object
# DESC  : Converts inputted expression string into a tree of Node objects.
# --------------------
def parse_exp(exp):

	oBracket = exp.find('(')
	cBracket = exp.rfind(')')

	if oBracket == -1 and cBracket == -1:
		return Node("const", [float(exp)])
	
	exp = split_exp(exp[oBracket+1:cBracket])

	vals = []
	for i in range(1, len(exp)):
		vals.append(parse_exp(exp[i]))

	return Node(exp[0], vals)

# --------------------
# FUNC  : split_exp
# RETURN: String
# DESC  : Partitions expression string by spaces not in brackets to isolate sub trees.
# --------------------
def split_exp(exp):
	b_count = 0
	breaks = [0]
	for i in range(0, len(exp)):
		if exp[i] == ' ' and b_count == 0: breaks.append(i)
		elif exp[i] == '(': b_count += 1
		elif exp[i] == ')': b_count -= 1
	return [exp[i:j] for i,j in zip(breaks, breaks[1:]+[None])]

# --------------------
# FUNC  : refresh_lambda
# RETURN: None
# DESC  : Sets inp and n parameters of lambda functions to use the data in evaluation of trees.
# --------------------
def refresh_lambda(inp, n):
	global operations
	operations.update({
		"data": lambda p, inp=inp, n=n: inp[abs(math.floor(p[0].evaluate()))%n] if n != 0 else 0,
		"diff": lambda p, inp=inp, n=n: inp[abs(math.floor(p[0].evaluate()))%n] - inp[abs(math.floor(p[1].evaluate()))%n] if n != 0 else 0,
		"avg" : lambda p, inp=inp, n=n: (lambda x, y, inp: (lambda i, j, inp: sum(inp[i:j])/(j-i))(min(x,y),max(x,y),inp) if x != y else 0)(abs(math.floor(p[0].evaluate()))%n, abs(math.floor(p[1].evaluate()))%n,inp) if n != 0 else 0
	})

# --------------------
# FUNC  : read_file
# RETURN: Two lists of Floats
# DESC  : Reads file and partitions data into two sets. One to evaluate and one to measure error. 
# --------------------
def read_file(filename):

	with open(filename) as f: data = f.readlines()

	data = [x.split('\t') for x in data]

	x = [[float(j.strip()) for j in i[:len(i)-1]] for i in data]
	y = [float(i[len(i)-1].strip()) for i in data]

	return x, y

# --------------------
# FUNC  : fitness
# RETURN: Float
# DESC  : Calculates the fitness of a tree using mean squared error.
# --------------------
def fitness(tree,x,y,m,n):

	try:

		fit = 0

		for i in range(0, m):
			inp = x[i]
			refresh_lambda(inp,n)
			fit += ( (y[i] - tree.evaluate()) ** 2) / m

		return fit
	
	except OverflowError: return float('inf')

# --------------------
# FUNC  : output_exp
# RETURN: String
# DESC  : Converts Node tree into string expression.
# --------------------
def output_exp(tree):

	if tree.op == "const": return str(tree.vals[0])

	param_string = ""
	for i in tree.vals:
		param_string += output_exp(i) + ' '
	param_string = param_string[:-1]

	return '(' + tree.op + ' ' + param_string + ')'

# --------------------
# FUNC  : generate_exp
# RETURN: Node object
# DESC  : Runs the genetic algorithm which generates a Node tree expression.
# --------------------
def generate_exp(pop_size, n, m, x, y, time_budget, max_depth=5, mutation_rate=0.5, elite=0.1, tourn_size=8, test_pop=None):

	elite = int(elite * pop_size)

	start = time.time()

	if test_pop == None:
		population = []
		for i in range(0, pop_size):
			new_tree = gen_random_tree(n, max_depth)
			new_tree.fitness = fitness(new_tree,x,y,m,n)
			population.append(new_tree)
	else:
		population = test_pop

	while(time.time() < start + time_budget):
		population = gen_new_pop(copy.deepcopy(population), pop_size, mutation_rate, elite, max_depth, tourn_size, x, y, m, n)

	population.sort(key=lambda x: x.fitness)
	
	return population[0]

# --------------------
# FUNC  : gen_random_tree
# RETURN: Node object
# DESC  : Generates a random Node tree.
# --------------------
def gen_random_tree(n, max_depth):

	if max_depth <= 2:

		if random.uniform(0, 1) < 0.3: return Node("const", [random.randint(0,n-1)], 1, 1)
		else:
			val = Node("const", [random.randint(0,n-1)], 1, 1)
			return Node("data", [val], 2, 2)

	op = random.choice(list(op_branches.keys()))

	branches = op_branches[op]

	vals = []
	size = 0
	depth = 0
	for i in range(0, branches):
		child = gen_random_tree(n, max_depth-1)
		vals.append(child)
		size += child.size
		if child.depth > depth: depth = child.depth

	return Node(op, vals, size + 1, depth + 1)

# --------------------
# FUNC  : sort_tree
# RETURN: Float
# DESC  : Used to sort objects using fitness and depth to prevent bloating.
# --------------------
def sort_tree(s, max_depth):
	if s.depth <= max_depth: return s.fitness
	else: return s.fitness * (1.3 ** (s.depth - max_depth))

# --------------------
# FUNC  : select
# RETURN: Two Node objects
# DESC  : Runs tournament selection on a given list of nodes.
# --------------------
def select(trees, max_depth, k=2):
	parents = random.sample(trees, k=k)
	parents.sort(key=lambda x: sort_tree(x,max_depth))
	return copy.deepcopy(parents[0]), copy.deepcopy(parents[1])

# --------------------
# FUNC  : crossover
# RETURN: Two Node objects
# DESC  : Takes two Node trees, selects a random branch from each and switches them.
# --------------------
def crossover(p1,p2):

	p1_branch, p1_path = choose_branch(p1)
	p2_branch, p2_path = choose_branch(p2)

	p1 = replace_branch(p1, p1_path, p2_branch)
	p2 = replace_branch(p2, p2_path, p1_branch)
	
	return p1, p2

# --------------------
# FUNC  : choose_branch
# RETURN: Node object and List of Ints
# DESC  : Selects a random branch from a given tree with equal probability. Returns branch and path to branch in original tree.
# --------------------
def choose_branch(tree, path = []):

	choice = random.uniform(0, 1)

	prob_inc = 1/tree.size

	if choice <= prob_inc: return tree, path

	prob_count = prob_inc

	for i in range(0, len(tree.vals)):
		prob_count += tree.vals[i].size * prob_inc
		if (choice <= prob_count):
			current_path = path + [i]
			return choose_branch(tree.vals[i], current_path)

# --------------------
# FUNC  : replace_branch
# RETURN: Node objects
# DESC  : Replaces a branch on a given Node tree, using the given path to the branch and given replacement tree.
# --------------------
def replace_branch(tree, path, replace):

	if path == []: return replace

	ind = path[0]
	tree.vals[ind] = replace_branch(tree.vals[ind], path[1:], replace)

	if tree.vals[ind].depth >= tree.depth: tree.depth = tree.vals[ind].depth + 1

	tree.size = 0
	for i in tree.vals: tree.size += i.size	

	return tree

# --------------------
# FUNC  : mutate
# RETURN: Node object
# DESC  : Selects a random branch, generates a random tree, and replaces the branch with the random tree.
# --------------------
def mutate(tree, n, max_depth):
	branch, path = choose_branch(tree) 
	branch = gen_random_tree(n, max_depth-len(path))
	return replace_branch(tree, path, branch)

# --------------------
# FUNC  : gen_new_pop
# RETURN: List of Node objects
# DESC  : Generates a new population for the genetic programming algorithm.
# --------------------
def gen_new_pop(parents, pop_size, mutation_rate, elite, max_depth, tourn_size, x, y, m, n):

	children = []
	children_count = 0

	while children_count < pop_size:

		p1, p2 = select(parents, max_depth, tourn_size)
		c1, c2 = crossover(p1, p2)

		if random.uniform(0, 1) < mutation_rate: c1 = mutate(copy.deepcopy(c1), n, max_depth)
		if random.uniform(0, 1) < mutation_rate: c2 = mutate(copy.deepcopy(c2), n, max_depth)

		c1.fitness = fitness(c1,x,y,m,n)
		c2.fitness = fitness(c1,x,y,m,n)

		children.append(c1)
		children.append(c2)
		children_count += 2

	parents.sort(key=lambda x: sort_tree(x,max_depth) )
	children.sort(key=lambda x: sort_tree(x,max_depth) )

	return parents[:elite] + children[:children_count-elite]

# --------------------
# FUNC  : out_tree
# RETURN: String
# DESC  : Prints tree for debugging.
# --------------------
def out_tree(tree, level = 0):

	tree_out = ('\t' * level) + "[" + tree.op + ", " + str(tree.depth) + "]\n"

	if tree.size > 1:
		for i in tree.vals: tree_out += out_tree(i, level+1)

	return tree_out

# ------------------------------------------------------------------------------------------------------------------------
# TESTING FUNCTIONS
# NOTE: THESE FUNCTIONS WILL NOT WORK WITHOUT A TIME SERIES FILE CALLED "output.dat" IN THE SAME DIRECTORY
# ------------------------------------------------------------------------------------------------------------------------

# --------------------
# FUNC  : test
# RETURN: None
# DESC  : Runs the algorithm 100 times given a set of paramters. Used to obtain the results in the Exercise 5 section "Testing Parameter Settings".
# --------------------
def test():

	iterations = 100

	pop_size = 90
	max_depth = 4
	mutation_rate = 0.3
	elite = 0.3
	tourn_size = 8
	time_budget = 10

	x, y = read_file("output.dat")
	n = 13
	m = 999
	
	results = []
	for i in range(0, iterations):
		progress(i, iterations, status='Running Test')
		t = generate_exp(pop_size, n, m, x, y, time_budget, max_depth, mutation_rate, elite, tourn_size)
		results.append(t.fitness)

	results.sort()

	low = results[0]
	q1 = results[24]
	med = results[49]
	q3 = results[74]
	high = results[99]
		
	sys.stdout.write('\033[K')
	sys.stdout.flush()

	print("-----------------------------------------------------")
	print("Max Depth:       ", max_depth)
	print("Mutation Rate:   ", mutation_rate)
	print("Elitism:         ", elite)
	print("Tournament Size: ", tourn_size)
	print("Population Size: ", pop_size)
	print("------------------------")
	print("Lowest Value:  ", low)
	print("Q1 Value:      ", q1)
	print("Median Value:  ", med)
	print("Q3 Value:      ", q3)
	print("Highest Value: ", high)
	print("-----------------------------------------------------")

# --------------------
# FUNC  : optimisingParameters
# RETURN: None
# DESC  : Runs the tests shown in the Exercise 5 chapter "Finding Optimum Parameters".
# --------------------
def optimisingParameters():

	# Static variables for testing
	pop_size = 30
	max_depth = 5
	mutation_rate = 0.5
	elite = 0.1 
	tourn_size = 2
	time_budget = 10

	x, y = read_file("output.dat")
	n = 13
	m = 999

	# TEST POPULATION
	population = []
	for i in range(0, 150):
		new_tree = gen_random_tree(n, max_depth)
		new_tree.fitness = fitness(new_tree,x,y,m,n)
		population.append(new_tree)
	
	# POPULATION TEST
	print("-----------------------------------------------------\n")
	test_results = "Population Test Results:\n"
	for i in range(30, 180, 30):
		average_fitness = test_parameters(i, time_budget, max_depth, mutation_rate, elite, tourn_size, "Testing Population Size: " + str(i), copy.deepcopy(population[:i]))
		test_results += "\t" + str(i) + ", " + str(average_fitness) +"\n"
	print(test_results)
	print("-----------------------------------------------------\n")

	# DEPTH TEST
	print("-----------------------------------------------------\n")
	test_results = "Depth Test Results:\n"
	for i in range(3, 11, 1):
		average_fitness = test_parameters(pop_size, time_budget, i, mutation_rate, elite, tourn_size, "Testing Depth Size: " + str(i), copy.deepcopy(population[:pop_size]))
		test_results += "\t" + str(i) + ", " + str(average_fitness) +"\n"
	print(test_results)
	print("-----------------------------------------------------\n")

	# MUTATION TEST
	print("-----------------------------------------------------\n")
	test_results = "Mutation Test Results:\n"
	for i in range(0, 11, 1):
		average_fitness = test_parameters(pop_size, time_budget, max_depth, i/10, elite, tourn_size, "Testing Mutation Rate: " + str(i/10), copy.deepcopy(population[:pop_size]))
		test_results += "\t" + str(i/10) + ", " + str(average_fitness) +"\n"
	print(test_results)
	print("-----------------------------------------------------\n")

	# ELITISM TEST
	print("-----------------------------------------------------\n")
	test_results = "Elite Proportion Results:\n"
	for i in range(0, 11, 1):
		average_fitness = test_parameters(pop_size, time_budget, max_depth, mutation_rate, i/10, tourn_size, "Testing Elite Proportion: " + str(i/10), copy.deepcopy(population[:pop_size]))
		test_results += "\t" + str(i/10) + ", " + str(average_fitness) +"\n"
	print(test_results)
	print("-----------------------------------------------------\n")

	# TOURNAMENT SELECTION TEST
	print("-----------------------------------------------------\n")
	test_results = "Tournament Test Results:\n"
	for i in range(2, 32, 2):
		average_fitness = test_parameters(pop_size, time_budget, max_depth, mutation_rate, elite, i, "Testing Tournament Size: " + str(i), copy.deepcopy(population[:pop_size]))
		test_results += "\t" + str(i) + ", " + str(average_fitness) +"\n"
	print(test_results)
	print("-----------------------------------------------------\n")

	# TIME BUDGET TEST
	print("-----------------------------------------------------\n")
	test_results = "Time Test Results:\n"
	for i in range(5, 45, 1):
		average_fitness = test_parameters(pop_size, i, max_depth, mutation_rate, elite, tourn_size, "Testing Time Budget: " + str(i), population[:pop_size])
		test_results += "\t" + str(i) + ", " + str(average_fitness) +"\n"
	print(test_results)
	print("-----------------------------------------------------\n")

# --------------------
# FUNC  : optimisingParameters
# RETURN: None
# DESC  : Worker function for the optimisingParameters function.
# --------------------
def test_parameters(pop_size, time_budget, max_depth, mutation_rate, elite, tourn_size, message, population):

	iterations = 100
	x, y = read_file("output.dat")
	n = 13
	m = 999

	average_fitness = 0
	for i in range(0, iterations):
		progress(i, iterations, status=message)
		t = generate_exp(pop_size, n, m, x, y, time_budget, max_depth, mutation_rate, elite, tourn_size, population)
		average_fitness += t.fitness

	sys.stdout.write('\033[K')
	sys.stdout.flush()

	return average_fitness / iterations

# --------------------
# FUNC  : progress
# RETURN: None
# DESC  : Progress bar output to console. Used in testing functions.
# --------------------
def progress(count, total, status=''):

	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))

	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)

	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
	sys.stdout.flush()

if __name__ == "__main__":
	main(sys.argv)





















