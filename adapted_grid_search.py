from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='train')
print("%d documents" % len(data.filenames))
print("%s categories" % (data.target_names))
print()