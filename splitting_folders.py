import split_folders
input='dataset'
output='final_dataset'
split_folders.ratio(input, output=output, seed=1337, ratio=(.8, .1, .1))
