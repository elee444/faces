import os
#DIR=os.listdir('/home/leee/Rubiks/faces/training_dataset/white')
DIR='/home/leee/Rubiks/faces/training_dataset'
for d in os.listdir(DIR):
    print(f"Before Renaming: {d}")    
    for f in os.listdir(DIR+"/"+d):
        os.chdir(DIR+"/"+d)
        if d not in f:
            os.rename(f, d+f[f.index('_'):])
            print(f"After Renaming: {d+f[f.index('0_'):]}")
        