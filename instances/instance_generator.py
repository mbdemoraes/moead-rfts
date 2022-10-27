import random

num_knaps = 3
for i in range(num_knaps):
  size=100
  profits=[random.randint(-50,50) for k in range(size)]
  filename = "bin_mucop_" + str(size) + "_" + str(i) + "_.txt"
  with open(filename, "a") as file:
    for p in profits:
      string = str(p) + "\n"
      file.write(string)