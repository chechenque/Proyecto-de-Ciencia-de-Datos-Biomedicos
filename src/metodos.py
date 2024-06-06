import random
juegos = ["Bully","Carmageddon","Dead Rising III","Fallout IV: New Vegas",
"Hearts of Iron","Max Payne","Hearts of Iron","Dead Island","Battlefield III","Call of Duty IV: Modern Warfare",
"Mass Effect","Mortal Kombat III","Postal"]

#print(random.randint(0,len(juegos)))

def test():
	
	juegosTest = []
	
	count = 0
	while count < 5:
		n = random.randint(1,len(juegos)-1)
		test1 = juegos[n]
		if not (test1 in juegosTest) :
			juegosTest.append(test1)
			count = count + 1
	return juegosTest

print(test())

		




	