from collections import Counter
import numpy as np
from PIL import Image
from numpy import asarray


def entropy_length(v):
    # Calculate the frequency of each element in the vector
    freq = Counter(v)

    # Create a list of tuples, where each tuple represents an element and its frequency
    tuples = [(element, frequency) for element, frequency in freq.items()]

    # Sort the list in ascending order by frequency
    tuples.sort(key=lambda x: x[1])

    # Build the Huffman tree
    while len(tuples) > 1:
        # Remove the two tuples with the lowest frequencies
        t1, t2 = tuples[0], tuples[1]
        tuples = tuples[2:]

        # Create a new tuple with the two tuples as elements and the sum of their frequencies as the frequency of the new tuple
        new_tuple = ((t1[0], t2[0]), t1[1] + t2[1])

        # Insert the new tuple back into the list, maintaining the sorted order
        tuples.append(new_tuple)
        tuples.sort(key=lambda x: x[1])

    # The resulting tuple is the root of the Haffman tree
    root = tuples[0][0]

    # Recursively traverse the tree and calculate the length of the Haffman code for each element
    lengths = {}

    def traverse(t, depth):
        if isinstance(t, tuple):
            # This is an internal node, so recursively traverse its children
            traverse(t[0], depth + 1)
            traverse(t[1], depth + 1)
        else:
            # This is a leaf node, so store the length of the Haffman code for this element
            lengths[t] = depth

    traverse(root, 0)

    return lengths


def entropy(v):
    # Calculate the frequency of each element in the vector
    freq = Counter(v)

    # Create a list of tuples, where each tuple represents an element and its frequency
    tuples = [(element, frequency) for element, frequency in freq.items()]

    entropy_bits = 0
    for (symbol, frequency) in tuples:
        probability = frequency / len(v)
        entropy_bits -= probability * np.log2(probability)

    print(entropy_bits)
    return entropy_bits


def numberOfBitsInEntropy(v):
    return len(v) * entropy(v)


def numberOfBitsInHuffman(v, lengths):
    count = 0
    for symbol in v:
        count += lengths[symbol]

    return count


v = [8, 14, 15, 1, 0, 8, 11, 14, 8, 4]
lengths = entropy_length(v)
print(numberOfBitsInHuffman(v, lengths))
print(numberOfBitsInEntropy(v))

# load the image
image = Image.open('Clown256B.bmp')
# convert image to numpy array
data = asarray(image)
data = data.reshape((1, -1))[0]

print(data)
lengths = entropy_length(data)
print(numberOfBitsInHuffman(data, lengths))
print(numberOfBitsInEntropy(data))

noise_picture = np.random.randint(0, 256, len(data))
lengths = entropy_length(noise_picture)
print(numberOfBitsInHuffman(noise_picture, lengths))
print(numberOfBitsInEntropy(noise_picture))
