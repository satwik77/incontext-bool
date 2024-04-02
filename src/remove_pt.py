import os
import argparse

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Remove all the .pt files in a directory')

	parser.add_argument('-folder_name', type=str, default='models', help='Name of the folder')

	return parser

def delete_pt_files(dir_path):
	for item in os.listdir(dir_path):
		item_path = os.path.join(dir_path, item)
		if os.path.isfile(item_path) and item_path.endswith('.pt'):
			os.remove(item_path)
		elif os.path.isdir(item_path):
			delete_pt_files(item_path)

if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	folname = args.folder_name

	delete_pt_files(folname)