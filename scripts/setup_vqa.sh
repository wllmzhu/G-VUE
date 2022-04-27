# Setup
mkdir ../datasets/data

# Get GloVe Embeddings
wget -P ../datasets/data http://nlp.stanford.edu/data/glove.6B.zip
unzip ../datasets/data/glove.6B.zip -d ../datasets/data/GloVe
rm ../datasets/data/glove.6B.zip