import sys
import os

#delta = 0.02
#i = 0 #4 #3 #2 #1
#pprs = [1.15 + (i * delta)]
#mcrs = [24.0]
#yx = 0

# using midpoints of the (-0.125,0.125,by=0.025) X (-22,22,by=1) relative to (1.15,24.0) figure
# all of these can exist on their own
#S1   (1.2125,35)  weak          UR quadrant "middle" in state space exploration
#S2   (1.1500,24)  reference     (0,0)
#S3   (1.0875,13)  middle        LL quadrant "middle"
#S4   (1.2125,13)  strong        LR quadrant "middle"
pprs = (1.2125,1.1500,1.0875,1.2125)
mcrs = (  35.0,  24.0,  13.0,  13.0)
s0 = 0  # weak
s1 = 1  # reference (0,0)
s2 = 2  # middle
s3 = 3  # strong

# change this to different strategy against itself (e.g., y1 vs y1, then y13a vs y13a)
strat1 = s0; strat2 = s1   # two weakest
#strat1 = s2; strat2 = s3   # two strongest
#strat1 = s0; strat2 = s3   # weakest vs strongest
#strat1 = s1; strat2 = s2   # two middle

## CHOOSE ONE AS TRUE!
#opts = (True,False,False)   # arr only
#opts = (False,True,False)   # div only
opts = (False,False,True)   # arr and div


NODES_PER_CPU = 28

ppr1_base = pprs[strat1]; mcr1_base = mcrs[strat1]
ppr2_base = pprs[strat2]; mcr2_base = mcrs[strat2] 

#seeds = [8675309]
#seeds = (5551212,2704357,5436611) # 1st 3 seeds
#seeds=(5551212,2704357,5436611,2435352,5303694,7883186,7927186,3416935,7057409,9809111) # 1st 10 seeds
#seeds=(2435352,5303694,7883186,7927186,3416935,7057409,9809111) # seeds 4-10
#seeds=(8616936,8709898,5006437,8544084,6249845,5598554,3341509,7745361,9263970,7447242,2856238,8352236,7442721,1545619,8623887,8620825,3993489,6036841,7025207,3291739) # last 20 seeds
#seeds=(2435352,5303694,7883186,7927186,3416935,7057409,9809111,8616936,8709898,5006437,8544084,6249845,5598554,3341509,7745361,9263970,7447242,2856238,8352236,7442721,1545619,8623887,8620825,3993489,6036841,7025207,3291739) # last 27 seeds
seeds=(5551212,2704357,5436611,2435352,5303694,7883186,7927186,3416935,7057409,9809111,8616936,8709898,5006437,8544084,6249845,5598554,3341509,7745361,9263970,7447242,2856238,8352236,7442721,1545619,8623887,8620825,3993489,6036841,7025207,3291739) # all 30 seeds

#seeds=(6075446,7501580,3209602,9771803,5565858,4085380,6150645,2274775,4193013,9312775,8283403,5548645,1085958,2718853,3893386,9930893,3804655,5114791,9772898,9442611,6926887,4895491,9859304,3505501,1736328,6971620,4170047,6529276,7576560,7675019) # diff 30 seeds

param_sets_written = 0
files_written = 0

def check(s: str) -> str: return "0.000" if s == "-0.000" else s

##########################################
# strategy #1
##########################################
ppr1 = ppr1_base
mcr1 = mcr1_base

##########################################
# strategy #2
##########################################
ppr2 = ppr2_base
mcr2 = mcr2_base

# for the figures in the Frontiers paper, we are modifying affinities only
# for the superior competitor... so:
#      (a) inferior @ (1.0,1.0)     superior @ (k, 1.0)  -- arrival only
#      (b) inferior @ (1.0,1.0)     superior @ (1.0, k)  -- division only
#      (c) inferior @ (1.0,1.0)     superior @ (k, k)    -- arrival & divsion
# where k \in (0.25, 0.5, 0.75, 1.0)
# affins = [0.25,0.50,0.75,1.00]
affins = [1.00]

arr_only    = opts[0]
div_only    = opts[1]
arr_and_div = opts[2]

try:
    assert(arr_only + div_only + arr_and_div == 1)
except:
    print("CHOOSE ONE OF ARR, DIV, or ARR&DIV")
    sys.exit(0)

arr_affin1 = 1.0
div_affin1 = 1.0

for affin in affins:

    if arr_only:
        arr_affin2 = affin
        div_affin2 = 1.0
    elif div_only:
        arr_affin2 = 1.0
        div_affin2 = affin
    else:
        arr_affin2 = affin
        div_affin2 = affin

    for seed in seeds:
        if param_sets_written % NODES_PER_CPU == 0:
            if files_written > 0:
                ofile.close()
            param_fname = f"param_space_files/param_space_{files_written}.txt"
            try:
                ofile = open(param_fname, "w")
                print(f"writing {param_fname}")
                files_written += 1
            except Exception as err:
                print(err)
                sys.exit(0)

        ofile.write(f"{seed} " + \
            f"{strat1} {ppr1} {mcr1} {arr_affin1} {div_affin1} " + \
            f"{strat2} {ppr2} {mcr2} {arr_affin2} {div_affin2}\n")
        param_sets_written += 1

ofile.close()
print(f"param sets written    = {param_sets_written:>7}")
print(f"param sets (w/o seed) = {param_sets_written//len(seeds):>7}")
