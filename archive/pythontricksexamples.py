# -*- coding: utf-8 -*-
"""
"""
#%% Simple repeater 
class Repeater:
  def __init__(self, value):
    self.value = value

  def __iter__(self):
    return RepeaterIterator(self)
    

class RepeaterIterator:
  def __init__(self, source):
    self.source = source
  
  def __next__(self):
    self.source.value

repeater = Repeater("Hello")

repeater

iterator = repeater.__iter__()

iterator

iterator.source

iterator.source.value

"""## Now with an end"""

my_list = [1,2,3]
my_list.__repr__()

object.__repr__(my_list)

iterator = my_list.__iter__()
iterator

iterator.__next__()

"""#### Class bounded Repeater"""

class BoundedRepeater:
  def __init__(self, value, max_repeats):
    self.value = value
    self.max_repeats = max_repeats
    self.count = 0

  def __iter__(self):
    return BoundedIterator(self)

  def __repr__(self):
    return f"Bounded with count = {self.count}"
  
class BoundedIterator:
  def __init__(self, source):
    self.source = source
  
  def __next__(self):
    if self.source.count >= self.source.max_repeats:
      raise StopIteration
    
    self.source.count = self.source.count + 1
    return self.source.value

repeater = BoundedRepeater("Hello", 3)
iterator = iter(repeater)

# %%

#%%
iterator.source

next(iterator)

iterator.source.count

iterator.source


# %% Generators

def bounded_repeater(value, max_repeats):
  print(f"Start of generator with value = {value}")
  count = 0
  while True:
    print(f"Start of loop with count = {count}") 
    if count < max_repeats:
      count += 1
      yield f"RETURNING {value} | count {count}"
    else:
      break
    
### I suspect that if the function has a yield statement
### python knows to make it a generator function (rather than normal)
### Calling it the first time only creates a generator object
### generator objects have the __next__() method
### Code inside the function doesn't actually RUN until we call __next__()
### Even when we "initialize" the first line doesn't run
### and thus the return of a generator is StopIteration


    # Generator function contains one or more yield statements.
    # When called, it returns an object (iterator) but does not start execution immediately.
    # Methods like __iter__() and __next__() are implemented automatically. So we can iterate through the items using next().
    # Once the function yields, the function is paused and the control is transferred to the caller.
    # Local variables and their states are remembered between successive calls.
    # Finally, when the function terminates, StopIteration is raised automatically on further calls.


class BoundedRepeater:
  def __init__(self, value, max_repeats):
    self.value = value
    self.count = 0
    self.max_repeats = max_repeats

  def __iter__(self):
    print(f"Start of loop with count = {self.count}")  
    return self

  def __next__(self):
    self.count += 1
    if self.count > self.max_repeats:
      raise StopIteration

    print (f"RETURNING {self.value} | count {self.count}") 
    return self.value


# %%
def bounded_repeater_for(value, max_repeats):
  print("Running first line of function")
  for i in range(max_repeats):
    yield f"{value} | count {i}"

# %%
var = BoundedRepeater("Hi", 3)
var_it = iter(var)

# for i in bounded_repeater("hi", 3):
#     print(i)

for i in bounded_repeater_for("hi", 3):
    print(i)


