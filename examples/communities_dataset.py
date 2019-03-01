import socialsim as ss


# Load some data
dataset = ss.load_data('data/debug_dataset.txt')

communities_directory = 'data/communities/'

dataset = ss.add_communities_to_dataset(dataset, communities_directory)

print(dataset[:10])
