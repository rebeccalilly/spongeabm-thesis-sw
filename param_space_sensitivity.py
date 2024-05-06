###########################################
# modified -- BUT NOT TESTED -- 25 Feb 2023
###########################################

import sys
#import os.path
#import glob
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


NODES_PER_CPU = 28

ppr1_base = pprs[strat1]; mcr1_base = mcrs[strat1]
ppr2_base = pprs[strat2]; mcr2_base = mcrs[strat2] 

################################################
##############################################
# use this when running for state space explor.
#
ppr_delta = 0.0025
ppr_change_lower = -0.125
#ppr_change_lower = 0.000
ppr_change_upper = 0.125

#ppr_delta = 0.001
#ppr_change_lower = 0.000
##ppr_change_upper = 0.050
###################################
mcr_delta = 1.0
##mcr_change_lower = -12
##mcr_change_upper = 12
mcr_change_lower = -22
mcr_change_upper = 22
#mcr_change_upper = 0

#mcr_delta = 0.5
#mcr_change_lower = -10
#mcr_change_upper = 5
###############################################
###############################################

#seeds = [8675309]
#seeds = (5551212,2704357,5436611) # 1st 3 seeds
#seeds=(5551212,2704357,5436611,2435352,5303694,7883186,7927186,3416935,7057409,9809111) # 1st 10 seeds
#seeds=(2435352,5303694,7883186,7927186,3416935,7057409,9809111) # seeds 4-10
#seeds=(8616936,8709898,5006437,8544084,6249845,5598554,3341509,7745361,9263970,7447242,2856238,8352236,7442721,1545619,8623887,8620825,3993489,6036841,7025207,3291739) # last 20 seeds
#seeds=(2435352,5303694,7883186,7927186,3416935,7057409,9809111,8616936,8709898,5006437,8544084,6249845,5598554,3341509,7745361,9263970,7447242,2856238,8352236,7442721,1545619,8623887,8620825,3993489,6036841,7025207,3291739) # last 27 seeds
seeds=(5551212,2704357,5436611,2435352,5303694,7883186,7927186,3416935,7057409,9809111,8616936,8709898,5006437,8544084,6249845,5598554,3341509,7745361,9263970,7447242,2856238,8352236,7442721,1545619,8623887,8620825,3993489,6036841,7025207,3291739) # all 30 seeds
#seeds=(5551212 2704357 5436611 2435352 5303694 7883186 7927186 3416935 7057409 9809111 8616936 8709898 5006437 8544084 6249845 5598554 3341509 7745361 9263970 7447242 2856238 8352236 7442721 1545619 8623887 8620825 3993489 6036841 7025207 3291739) # all 30 seeds

param_sets_written = 0
files_written = 0

def check(s: str) -> str: return "0.000" if s == "-0.000" else s

##########################################
# strategy #1
##########################################
ppr1 = ppr1_base + ppr_change_lower
ppr1 = float(f"{ppr1:0.4f}")
while ppr1 <= ppr1_base + ppr_change_upper:

    ppr1 = float(f"{ppr1:0.4f}")
    ppr1_diff = ppr1 - ppr1_base

    mcr1 = mcr1_base + mcr_change_lower
    mcr1 = float(f"{mcr1:0.3f}")
    while mcr1 <= mcr1_base + mcr_change_upper:

        mcr1 = float(f"{mcr1:0.3f}")
        mcr1_diff = mcr1 - mcr1_base

        arr_affin1 = 1.0
        div_affin1 = 1.0

        ##########################################
        # strategy #2
        ##########################################
        #ppr2 = ppr2_base - ppr_half_width
        #while ppr2 <= ppr2_base + ppr_half_width:
        ppr2 = ppr2_base
        for yy in range(1):   # strategy #2 stays fixed -- control

            #ppr2_diff = ppr2 - ppr2_base
            ppr2_diff = 0.0

            #mcr2 = mcr2_base - mcr_half_width
            #while mcr2 <= mcr2_base + mcr_half_width:
            mcr2 = mcr2_base
            for zz in range(1):   # strategy #2 stays fixed -- control

                #mcr2_diff = mcr2 - mcr2_base
                mcr2_diff = 0.0

                arr_affin2 = 1.0
                div_affin2 = 1.0

                pairing1 = (f"({ppr1_diff:.4f},{mcr1_diff:.3f},{arr_affin1},{div_affin1}) " + \
                            f"({ppr2_diff:.4f},{mcr2_diff:.3f},{arr_affin2},{div_affin2})")
                pairing2 = (f"({ppr2_diff:.4f},{mcr2_diff:.3f},{arr_affin2},{div_affin2}) " + \
                            f"({ppr1_diff:.4f},{mcr1_diff:.3f},{arr_affin1},{div_affin1})")

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

                    #print(f"{s} " + \
                    #    f"{strat1} {ppr1} {ppr1_diff:.4f} {mcr1} {mcr1_diff:.3f} {arr_affin1} {div_affin1} " + \
                    #    f"{strat2} {ppr2} {ppr2_diff:.4f} {mcr2} {mcr2_diff:.3f} {arr_affin2} {div_affin2}")
                    ppr1_diff_f3 = check(f"{ppr1_diff:.4f}")  # look for "-0.000" (ugh)
                    mcr1_diff_f3 = check(f"{mcr1_diff:.3f}")
                    ppr2_diff_f3 = check(f"{ppr2_diff:.4f}")
                    mcr2_diff_f3 = check(f"{mcr2_diff:.3f}")
                    ofile.write(f"{seed} " + \
                        f"{strat1} {ppr1} {ppr1_diff_f3} {mcr1} {mcr1_diff_f3} {arr_affin1} {div_affin1} " + \
                        f"{strat2} {ppr2} {ppr2_diff_f3} {mcr2} {mcr2_diff_f3} {arr_affin2} {div_affin2}\n")
                    param_sets_written += 1
                    #print(f"{s} " + \
                    #    f"{strat1} {ppr1} {ppr1_diff:.4f} {mcr1} {mcr1_diff:.4f} {arr_affin1} {div_affin1} " + \
                    #    f"{strat2} {ppr2} {ppr2_diff:.4f} {mcr2} {mcr2_diff:.4f} {arr_affin2} {div_affin2}")

            #mcr2 += mcr_delta

        #ppr2 += ppr_delta
        ##########################################
    mcr1 += mcr_delta
    ##########################################
ppr1 += ppr_delta
##########################################

ofile.close()
print(f"param sets written    = {param_sets_written:>7}")
print(f"param sets (w/o seed) = {param_sets_written//len(seeds):>7}")
