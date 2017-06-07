GLOB_PREDICTIONS = "ensembling/*"
RK_AVG_OUTPUT = "rk_avg_ensemble.csv"

from collections import defaultdict
from glob import glob
import sys

def rk_avg_ensemble(glob_files, loc_outfile):
  # write to the chosen file
  with open(loc_outfile,"w") as outfile:
    # store the raw ranks
    # keys are in the form ()
    all_ranks = defaultdict(list)
    # i is needed to write the header of the output file
    for i, glob_file in enumerate( glob(glob_files) ):
      # define a container for the ranks found in a file
      file_ranks = []
      print("Parsing now: ", glob_file)
      # sort glob_file by first column, ignoring the first line
      lines = open(glob_file).readlines()
      lines = [lines[0]] + sorted(lines[1:])
      # for each line, store the line number and the line itself
      for e, line in enumerate( lines ):
        if e == 0 and i == 0:
          outfile.write( line )
        # if the line is not a header, process it
        elif e > 0:
          # store the row in a list
          row = line.strip().split(",")
          # <row[0]> is the predicted probability
          # <e> is the line number
          # <row[1]> is the id of the question
          file_ranks.append( (float(row[0]), e, row[1]) )
      # sort by the predicted probability
      # and give it a rank
      for rank,container in enumerate( sorted(file_ranks) ):
        # store the rank in the dictionary for further sorting
        # the key is in the form (<line number>, <id>)
        all_ranks[(container[1],container[2])].append(rank)
        
    # define a list as a container for the average ranks
    average_ranks = []
    # sort by the line number
    # k is in the form (<line number>, <id>)
    for identifier in sorted(all_ranks):      
      # append the average rank together with the identifier
      average_ranks.append((sum(all_ranks[identifier])/len(all_ranks[identifier]),
                            identifier))
    # define a list as a container for the ranked ranks
    sorted_ranks = []
    # the element of the <average_ranks> is in the format
    # (<average_rank>, <identifying_container>), where
    # <identifying_container> is (<line number>, <id>)
    
    for rank, avg_rk_obj in enumerate(sorted(average_ranks)):
      # sort <average_ranks> by the average rank and
      # append (<line number>, <id>, <normalized probability>)
      # to <sorted_ranks> 
      sorted_ranks.append((avg_rk_obj[1][0],
                           avg_rk_obj[1][1],
                           rank/(len(average_ranks)-1)))
    # norm_prob_obj is in the format
    # (<line number>, <id>, <normalized probability>)
    for norm_prob_obj in sorted(ranked_ranks):
      outfile.write("{},{}\n".format(norm_prob_obj[2],
                                     norm_prob_obj[1]))
    print("Saved the normalised probabilites to {}.".format(loc_outfile))

rk_avg_ensemble(GLOB_PREDICTIONS, RK_AVG_OUTPUT)
