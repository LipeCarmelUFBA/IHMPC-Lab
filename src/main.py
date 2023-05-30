import os
print()
here = os.path.abspath('Examples')
#runfile('C:/Users/lipec/Documents/GitHub/IHMPC-Lab/src/Examples/Stable Systems/Set-point tracking/DistillationColumn.py', wdir=here)
for root, dirs, files in os.walk(here):
    for file in files:
        if file.endswith('.py'):
            runfile(os.path.join(root, file), wdir=here)