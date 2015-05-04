#!/usr/bin/env python3

from sound import *
import random
import time
from eigenvalues import arrange_eigenvalues

# A small game using sound.py

def sound_game():
    eigenvalues = arrange_eigenvalues(pickle.load(open('eigenvalues/eigenvalues.db', "rb")))
    dif = int(input('Enter the difficulty (2-10): ')) # number of sounds to heard
    category = random.choice(list(eigenvalues.keys())) # a random category
    sample = random.randint(0,len(eigenvalues[category])-1) # a random object in this category
    print("Listen this sample of " + category)
    time.sleep(2)
    produce_sound(eigenvalues[category][sample][1])
    print("Now listen these " + str(dif) + " sounds:")
    now = random.randint(0,dif-1) # time to listen an object of the same category than sample
    for i in range(0,dif):
      time.sleep(2)
      print("Sound " + str(1+i))
      if i == now: # a random object in the same category than sample
        produce_sound(eigenvalues[category][random.randint(0,len(eigenvalues[category])-1)][1])
      else:
        rand_cat = random.choice(list(eigenvalues.keys()))
        while rand_cat == category:
          rand_cat = random.choice(eigenvalues.keys())
        produce_sound(eigenvalues[rand_cat][random.randint(0,len(eigenvalues[rand_cat])-1)][1]) # a random object not in the category of sample
    answer = int(input("Which one of these sounds was %s?\n" % category))
    if answer-1 == now:
        print("Success")
    else:
        print("Fail, it was number " + str(now+1))

if __name__ == "__main__":
    sound_game()
