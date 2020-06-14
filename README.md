# Query-by-Humming 
(Part of EE392A UGP)

First of all unzip all the folders.

To use yin, yinfft, COMB filters, schmitt, specacf-
  1. Prepare the note sequences by passing desired method and folder name as arguments. eg- "python aub_method_pitch_ext.py --dataset hummed --method mcomb"
  2. To get a ranked list of indices run "aub_method_matcher.py" with target and hummed dataset as arguments. eg-"python aub_method_matcher.py --tar_dataset target --hum_dataset=hummed"

To make plot of notes vs time-
  1. Pass the desired folder and method as arguments and run. eg- "python notes_plotter.py --dataset target --method mcomb"
