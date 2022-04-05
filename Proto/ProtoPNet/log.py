import os

def log(line, file):
  with open(path + file, 'a+') as log:
      content = log.read()
      log.write(content + line + str("\n"))
