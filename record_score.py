from tensorboardX import SummaryWriter
import numpy as np
import os
import time

writer_dir = '/data/dongchengbo/tianchi_output/all_scores'
writer = SummaryWriter(logdir=writer_dir)

score_dict = {}

while True:
    draw = []
    for root,dirs,files in os.walk(writer_dir):
        for file in files:
            if '.txt' in file:
                path = os.path.join(root,file)
                name = file.split('.')[0]
                with open (path,'r') as f:
                    temp = f.readlines()
                    temp = [float(each.strip()) for each in temp]


                if name in score_dict:
                    old_v = score_dict[name]
                    score_dict[name] = temp
                    if len(old_v) == len(temp):
                        print("%s not change"%name)
                        continue
                    else:
                        start = len(old_v)
                        print("%s add %d item" %(name, len(temp) - start))
                        draw.append((name, start, temp[start:]))

                else:
                    score_dict[name] = temp
                    start = 0
                    print("%s add %d item" % (name, len(temp) - start))
                    draw.append((name, start, temp))


    for each in draw:
        name, start, values = each
        for ix in range(len(values)):
            writer.add_scalars("all_scores", {name: values[ix]}, start + ix)
    print("-----------------------")
    time.sleep(2000)

