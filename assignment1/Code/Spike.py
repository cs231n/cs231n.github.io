import os

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as f:
        dict = cPickle.load(f, encoding='bytes')

    return dict

if __name__ == '__main__':
    CURR_DIR = os.getcwd()
    os.chdir('/Users/jnarhan/Projects/Stanford_CNN_Course/assignment1/cs231n/datasets/cifar-10-batches-py/')
    print(os.getcwd())
    imgs = unpickle('data_batch_1')
    print(imgs)

