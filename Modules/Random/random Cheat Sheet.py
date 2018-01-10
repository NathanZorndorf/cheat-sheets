import random

#---- Generate a seed : After a random seed is set, the numbers generated after will follow the same sequence.
random.seed(a=None)

#---- Generate a random integer 
random.randint(0, 10)
	# Example : make a list of random integers :
	new_sequence = [random.randint(0,10) for num in range(0,10)]

#---- Select random elements from a list 
random.sample(population, k) # returns a k length list of unique elements chosen from population sequence 
	# Example : choose a sample from a range of integers:
	random.sample(range(10000000), 60)


