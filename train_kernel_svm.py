#!/usr/bin/env python3
"""
Multiclass text classification with kernel-SVM with intersection kernel

Memory requirement: ~O(N^2) memory, where N is the number of documents
"""

import os
import sys
import csv
import warnings
import re
import pandas
from sklearn import svm
from pymorphy2 import MorphAnalyzer
import numpy as np
from collections import defaultdict
from random import randint
import pickle
from tqdm import tqdm
from intersection import kernel
np.random.seed(1)

from scipy.sparse import csr_matrix

NORMAL = {}  # unlimited cache: word -> normal_form
PARSER = MorphAnalyzer()

MAX_PER_CLASS = 350

stop_words = "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между"
stop_words = stop_words.split()
set_stop_words = set(stop_words)

def get_normal(word):
    """ Convert word to normal form """
    if word not in NORMAL:
        NORMAL[word] = PARSER.parse(word)[0].normal_form
    return NORMAL[word]

def text2words(text):
    """Preprocess text and split into words"""

    # Some domain-specific preprocessing
    i = text.find("РИА Новости")
    if 0 <= i <= 30:
        text = text[i + len("РИА Новости."):]
    text = text.replace("Поделиться", "")

    # Sometimes spaces are not added between parts of text, so add space before first
    #  uppercase letter in a row
    set_upper = set(list("QWERTYUIOPASDFGHJKLZXCVBNMЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЁЯЧСМИТЬБЮ"))
    new_text = []
    prev_upper = False
    for c in text:
        if c in set_upper:
            if prev_upper:
                new_text.append(c)
            else:
                new_text.append(" " + c)
            prev_upper = True
        else:
            prev_upper = False
            new_text.append(c)
    text = "".join(new_text)
    
    # Make lowercase
    text = text.lower()
    # Replace special symbols by space
    text = re.sub('[,.:;"!?«»]', ' ', text)
    # Only leave allowed symbols
    set_allowed = set(list("%qwertyuiopasdfghjklzxcvbnmйцукенгшщзхъфывапролджэёячсмитьбю' "))
    text = "".join(c for c in text if c in set_allowed)
    # Convert words to normal form
    words = []
    for w in text.split():
        if w == '%':
            words.append(w)
        else:
            words.append(get_normal(w))
    # Remove stop words
    words = [w for w in words if w not in set_stop_words]
    return words

def words2bow(text, voc):
    """ Convert a list of words into BoW descriptor """
    desc = np.zeros(len(voc))
    for w in text:
        if w in voc:
            desc[voc[w]] += 1
    
    # Actually bigrams follow a different distribution, maybe some other normalization would be better
    # Still, in this way they slightly help
    for i in range(len(text)-1):
        bigram = text[i] + " " + text[i+1]
        if bigram in voc:
            desc[voc[bigram]] += 1
    return desc

def dataset2bows(data, voc=None):
    """ Convert a set of texts into BoW descriptors 
    data - documents (list of str)
    voc - vocabulary (None or dict: word -> index)
    """

    # Convert text to words
    docs = []
    words = []
    for i in tqdm(range(len(data))):
        docs.append(text2words(data[i]))
        if voc is None:
            words.extend(docs[-1])
    
    # Build vocabulary
    if voc is None:
        voc = defaultdict(int)
        
        for w in words:
            voc[w] += 1
        
        # Add bigrams (homework: identify a minor problem here)
        for i in range(len(words)-1):
            voc[words[i] + " " + words[i+1]] += 1

        print("Total words: ", len(voc))

        voc = [w for w, cnt in voc.items() if cnt > 3]
        voc = {w:i for i, w in enumerate(voc)}

        print("After removing rare: ", len(voc))

        del words

    # Compute descriptors
    descs = np.zeros((len(docs), len(voc)))
    for i in tqdm(range(len(docs))):
        descs[i] = words2bow(docs[i], voc)
    
    return voc, csr_matrix(descs)

# Unomptimized dense version - kernel between 2 vectors
#def intersection_kernel(x1, x2):
#    return np.sum(np.minimum(x1, x2))

def intersection_kernel(A, B):
    """ A: [n1 X d]
        B: [n2 X d]
    Returns:
        K: [n1 X n2]
    Intersection kernel matrix precomputation for scipy.sparse.csr_matrix'es

    See README on how to compile the Cython code
    """
    
    # Not sure, if we could have used float32 here - see how sklearn/libsvm is implemented

    if not isinstance(A, csr_matrix):
        A = csr_matrix(A, dtype=np.float64)
    A.sort_indices()
    if not isinstance(B, csr_matrix):
        B = csr_matrix(B, dtype=np.float64)
    B.sort_indices()
    
    K = np.zeros((A.shape[0], B.shape[0]))
    
    kernel(A.shape[0], B.shape[0], A.indptr, B.indptr, A.indices, B.indices, A.data, B.data, K)

    return K

def calc_acc_binary(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    return 100 * np.sum(np.sign(scores) == labels) / float(labels.size)

def select_fold(x, all_val_inds, ifold):
    """
    Select i-th fold of n for cross validation from array x of data/labels
    The first dimension of x must correspond to instances.
    x must be numpy.array
    """
    assert(isinstance(x, np.ndarray))
    assert(isinstance(all_val_inds, list))

    from functools import reduce
    val_inds = all_val_inds[ifold][:]
    val = x[val_inds].copy() 

    train_inds = all_val_inds[:ifold] + all_val_inds[ifold+1:]
    train_inds = reduce(lambda a,b: a+b, train_inds)
    train = x[train_inds].copy()
    return (train, val, train_inds, val_inds)

def train_binary(trainlabels, K, cost=None, weights=None, crit='ap', n_folds=5,
    verbose=False, crv_path=None, store_training=False, avg='median',
    seed=1, multilabels=None):
    """
    Train a binary kernel-SVM classifier given a precomputed kernel K
    This code was originally written in Matlab and later translated into Python

    Many thanks to Adrien Gaidon, Dan Oneata who helped me to write and improve this function. 

    trainlabels - +1 / -1
    K - precomputed kernel
    cost - exact value _or_ list of C parameters (None => use default range)
    weights - weights for each class - dict {class: weight} (None => use default balancing)
    crit - criterium to select best C ('ap', 'acc' or 'ba')
    n_folds - number of cross-validation folds
    avg - way to average fold performances ('median' or 'mean')
    multilabels - multiclass labels in order to make stratified split for cross-validation
    
    verbose - if True, increases verbosity
    crv_path - path to an image to save a cross-validation plot
    seed - random seed (if 0 - use random random seed)
    
    Returns:
        model - trained model
        val_perfs - array of preformances [n_folds X len(vc)]
        ibest - index of the best C
        val_scores - list of length n_folds of crossval. scores for the best C
        val_labels - list of length n_folds of labels for each fold
    """

    trainlabels = np.ravel(np.array(trainlabels))

    if weights is None:
        weights = {1: (1.0 / np.sum(trainlabels > 0)), \
            -1: (1.0 / np.sum(trainlabels < 0))}
    else:
        assert(isinstance(weights, dict))
        assert(set(weights.keys()) == set([-1, 1]))

    if verbose:
        print('===train_classifiers===')

        print('trainlabels: %d pos, %d neg' % (np.sum(trainlabels>0), np.sum(trainlabels<0)))
    
    def svm_opts(weights):
        return {'kernel':'precomputed', 'class_weight':weights,
            'probability':True}
    
    # Convert scalar to list (in case of exact value)
    #   (cost.size == 1 in case of np.float64 type)
    if (not hasattr(cost, 'size') or cost.size == 1) and cost != None and \
        not isinstance(cost, list):
        cost = [cost]

    # Prepare list of C for grid search
    if cost != None:
        vc = list(cost)
    else:
        vc = [(10**i) for i in np.arange(-2, 6+0.1, 0.5)]  # vector of C values
    
    if seed == 0:
        seed = None
    random_state = np.random.RandomState(seed)

    print("n_folds=", n_folds)
    val_perfs = np.zeros((n_folds, len(vc)))  # validation performances (see 'crit')

    p = random_state.permutation(len(trainlabels))
    fold_inds = []
    if multilabels is None:
        # Non-stratified split
        for i in range(n_folds):
            fold_inds.append(list(p[i*len(trainlabels)//n_folds:
                                    (i+1)*len(trainlabels)//n_folds]))
    else:
        # Stratified split
        multilabels = np.array(multilabels)
        for i in range(n_folds):
            fold_inds.append([])
        classes = sorted(set(multilabels))
        for y in classes:
            class_inds = np.flatnonzero(multilabels[p] == y)
            for i in range(n_folds):
                fold_inds[i].extend(p[class_inds[i*len(class_inds)//n_folds:
                                                 (i+1)*len(class_inds)//n_folds]])
        assert(set([x for fi in fold_inds for x in fi]) == set(range(len(trainlabels))))

    all_val_scores = [[] for i in range(n_folds)]
    all_val_labels = []

    if verbose:
        print('Crossvalidating for C in [%0.3f, %0.3f] - %d values' % (min(vc), max(vc), len(vc)))
    for ic in range(len(vc)):
        for i in range(n_folds):
            # Prepare folds for training and validation
            (labels, vallabels, inds, valinds) = select_fold(trainlabels,
                fold_inds, i)
            labels = np.array(labels)

            assert(np.sum(labels > 0) > 0)
            assert(np.sum(labels < 0) > 0)
            
            weights_ic = {1: (1.0 / np.sum(labels > 0)), \
                -1: (1.0 / np.sum(labels < 0))}
            # Construct classifiers                
            clf = svm.SVC(C=vc[ic], **(svm_opts(weights_ic)))
            train_ker = K[inds, :][:, inds]

            clf.fit(train_ker, labels)

            val_ker = K[valinds, :][:, inds]           
            
            #scores = clf.decision_function(val_ker)
            
            # Warning: May be too inaccurate to use probs during cross-validation!
            scores = clf.predict_proba(val_ker)[:, 1]

            all_val_scores[i].append(scores)
            if ic == 0:
                all_val_labels.append(vallabels)

            # Get predicted labels
            results = clf.predict(val_ker)

            if crit == 'ap':
                val_perfs[i, ic] = calc_ap_exact(scores, vallabels) # use scores instead of probs!
            elif crit == 'acc':
                val_perfs[i, ic] = calc_acc_binary(results, vallabels)
            elif crit == 'ba':
                val_perfs[i, ic] = calc_b_acc(results, vallabels)

            if (np.sum(vallabels > 0) == 0 or np.sum(vallabels < 0) == 0):
                print("Only 1 vallabel:", set(vallabels), end='')
                print("Perf = %0.3f" % val_perfs[i, ic])

    # Print performances
    if verbose:
        for ic in range(len(vc)):
            print('C = %8g -> %8g +/- %8g' % \
                (vc[ic], np.mean(val_perfs[:, ic]), np.std(val_perfs[:, ic])))

    if avg == 'median':
        warnings.warn("Using median averager.")
        mean_crit = np.median(val_perfs, 0)
    elif avg == 'mean':
        mean_crit = np.mean(val_perfs, 0)
    else:
        raise KeyError('Unknown averager: %s' % avg)

    # Find all cases that achieve the best performance
    candidates = (mean_crit > 0.9999*np.max(mean_crit))
    if verbose and (np.sum(candidates) > 1):
        print('plato in cross-validation curve (criterium %s)' % crit)
    v_ibests = np.flatnonzero(candidates)  # vector of best indices

    # Choose leftmost candidate
    ibest = np.min(v_ibests)

    # When plateau on the left, select its right-most point
    if ibest == 0 and len(vc) > 1: 
        print("Plateau on the left")
        ibest = None
        for i in range(1, len(mean_crit)):
            if i not in v_ibests:
                ibest = i - 1
                break
        if ibest == None:
            print("WARN: absolute plateau")
            ibest = len(mean_crit) - 1

    cost = vc[ibest]
    if verbose:
        print('C = %f' % cost)
    
    if verbose or crv_path:
        # Save grid-search plot (aka cross-validation curve)
        if crv_path == None:
            import tempfile
            file = tempfile.NamedTemporaryFile(delete=False)
            file.close()
            os.remove(file.name)
            crv_path = file.name + '.png'

        if crit == 'ap':
            npos = np.sum(trainlabels > 0)
            n = np.sum(trainlabels>0) + np.sum(trainlabels<0)
            baseline = 100*float(npos)/n
        elif crit == 'ba':
            baseline = 50.0
        else:
            baseline = None
        
        crv_plot(crv_path, cost, vc, val_perfs, baseline)
        
        print("Cross-validation plot written to", crv_path)

    # Train the final classifier on the whole data
    clf = svm.SVC(C=cost, **(svm_opts(weights))) 
    clf = clf.fit(K, trainlabels)
    model = {'clf':clf}
    
    val_scores = [x[ibest] for x in all_val_scores]
    val_labels = all_val_labels
    return (model, val_perfs, ibest, val_scores, val_labels)

def crv_plot(save_file, c_best, vc, val_perfs, baseline=None):
    """ Grid-search plot (aka cross-validation curve) """
    import matplotlib as mp
    mp.use('Agg') #, warn=False)
    import matplotlib.pyplot as plt
    plt.ioff()
    
    # Show cross-validation plot
    plt.figure()
    plt.clf()
    mvp = np.mean(val_perfs, 0)
    plt.errorbar(vc, mvp, yerr=[mvp-np.min(val_perfs, 0), np.max(val_perfs, 0)-mvp], capsize=3,
                 label='Min-Max', color='black')
    plt.errorbar(vc, mvp, yerr=np.std(val_perfs, 0), color='green', linewidth=2, capsize=5,
                 label='Standard deviation')
    plt.xscale('log')
    plt.ylim([0, 100])
    plt.xlim([vc[0], vc[-1]])

    plt.plot(c_best, mvp[np.array(vc)==c_best], 'or', label='Chosen C')

    if baseline:
        plt.plot([vc[0], vc[-1]], [baseline, baseline], 'b--', label='Baseline')
    
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig(save_file)

def test_binary(K, model):
    """ K = [n_test X n_train] """
    Ki = np.ascontiguousarray(K[:, model['mask']])
    prob_scores = model['clf'].predict_proba(Ki)
    prob_scores = prob_scores[:, 1]
    scores = model['clf'].decision_function(Ki)
    results = model['clf'].predict(Ki)
    
    return list(results), list(scores), list(prob_scores)

def train(descs, labels, complementary_labels):
    """ 
    Train binary classifiers using "1 vs rest" strategy

    complementary_labels - tried to train 2 separate classifiers for classes "society"/"Russia",
    but this does not help"""

    print("Computing kernel")
    n = descs.shape[0]
    
    K = intersection_kernel(descs, descs)
    
    nc = max(labels) + 1
    models = [None]*nc

    mean_perfs = []
    std_perfs = []
    val_scores = [None]*nc
    val_labels = [None]*nc
    for i in range(nc):
        print("Class", i)
        binary_labels = [1 if y == i else (0 if (i in complementary_labels and y in complementary_labels) else -1) for y in labels]
        binary_labels = np.array(binary_labels)
        mask = (binary_labels != 0)
        Ki = np.ascontiguousarray(K[mask, :][:, mask])
        binary_labels = binary_labels[mask]
        models[i], val_perfs, i_best, _val_scores, _val_labels = train_binary(binary_labels, Ki, crit='acc', avg='mean', verbose=True, multilabels=labels)
        val_scores[i] = np.concatenate(_val_scores, axis=0)
        val_labels[i] = np.concatenate(_val_labels, axis=0)
        models[i]['mask'] = mask
        mean_perfs.append(np.mean(val_perfs[:, i_best]))
        std_perfs.append(np.std(val_perfs[:, i_best]))
        print("Accuracy = {:0.1f} +/- {:0.1f}".format(mean_perfs[-1], std_perfs[-1]))

    print("Binary classifier evaluation:")
    for i in range(nc):
        print("Class", i)
        print("Accuracy = {:0.1f} +/- {:0.1f}".format(mean_perfs[i], std_perfs[i]))

    print("Average accuracy = {:0.1f} +/- {:0.1f}".format(np.mean(mean_perfs), np.std(mean_perfs)))
    return models, val_scores, val_labels

def l1norm_it(descs, eps=1e-6):
    """ Normalize descriptors in-place """
    if isinstance(descs, np.ndarray):
        sums = np.abs(descs).sum(1)
        sums = np.asarray(sums) # sparse matrices return np.matrix
        valid = (sums > eps)
        if np.all(valid):
            descs[:] = descs[:] / sums[:, np.newaxis]
        else:
            descs[valid, :] = descs[valid, :] / sums[valid][:, np.newaxis]
    elif isinstance(descs, csr_matrix):
        for i in range(descs.shape[0]):
            s = descs.data[descs.indptr[i]:descs.indptr[i+1]].sum()
            if s > eps:
                descs.data[descs.indptr[i]:descs.indptr[i+1]] /= s
    else:
        assert False

def normalize_it(bofs, idf=None):
    """ Normalize descriptors in-place + take into account inverse document frequencies """
    
    if idf is not None:
        for i in range(len(bofs.indices)):
            bofs.data[i] *= idf[bofs.indices[i]]
    
    l1norm_it(bofs)

def load_test_voc_and_bows():
    voc_path = "data/voc4.pickle"
    test_bows_path = "data/test_bows4.pickle"
    if os.path.exists(voc_path) and os.path.exists(test_bows_path):
        print("Loading vocabulary and test bofs")
        with open(voc_path, 'rb') as f:
            voc = pickle.load(f)
        with open(test_bows_path, 'rb') as f:
            test_bows = pickle.load(f)
        #test_bows = test_bows.toarray()
    else:
        print("Loading test data")
        test_data = []
        with open("data/test_news.csv", encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                test_data.append(row[0])
        test_data = test_data[1:]

        voc, test_bows = dataset2bows(test_data)
        del test_data
        
        with open(voc_path, 'wb') as fw:
            pickle.dump(voc, fw)

        with open(test_bows_path, 'wb') as fw:
            pickle.dump(test_bows, fw)
    
    return voc, test_bows

if __name__ == '__main__':
    # Main part
    import sys
    csv.field_size_limit(sys.maxsize)

    # Load/compute test vocabulary and BOWs
    voc, test_bows = load_test_voc_and_bows()    
    # See further comments in print strings

    
    print("Computing idf")
    doc_counts = np.zeros(test_bows.shape[1], dtype=int)
    for i in range(test_bows.shape[0]):
        inds = list(set(test_bows.indices[test_bows.indptr[i]:test_bows.indptr[i+1]]))
        doc_counts[inds] += 1 
    doc_counts[doc_counts == 0] = 1
    idf = np.log(test_bows.shape[0] / doc_counts)

    print("Normalizing")
    normalize_it(test_bows, idf)
    
    print("Feature dimensions:", test_bows.shape)

    print("Loading data")
    df = pandas.read_pickle('data/train1.p', compression='gzip')
    print(df.topic.value_counts())
    
    print(f"Selecting at most {MAX_PER_CLASS} examples per class")
    classes = sorted(set(df.topic))
    map_classes = {y:i for i, y in enumerate(classes)}
    labels = [map_classes[y] for y in df.topic]
    labels = np.array(labels)
    inds = []
    for y in range(len(classes)):
        class_inds = np.flatnonzero(labels == y)
        class_inds = np.random.permutation(class_inds)
        if classes[y] not in ['society', 'Силовые структуры']:
            class_inds = class_inds[:MAX_PER_CLASS]
        inds += list(map(int, class_inds))

    df = df.iloc[inds]

    print("Total examples:", len(df))
    print(df.topic.value_counts())

    print("Computing BOWs")
    train_data = [x for x in df.content]
    _, train_bows = dataset2bows(train_data, voc)
    
    print("Normalizing")
    normalize_it(train_bows, idf)
    train_bows = csr_matrix(train_bows)

    print("Feature dimensions:", train_bows.shape)

    print("Training classifiers")
    classes = list(sorted(set(df.topic)))
    print(classes)
    labels = [classes.index(x) for x in df.topic]
    complementary_labels = [] #[classes.index('society'), classes.index('Россия')]

    models, val_scores, val_labels = train(train_bows, labels, complementary_labels)

    # Save cross-validation results (if not using complementary labels, val_scores match for all classes)
    with open("data/crv.pickle", 'wb') as fw:
        pickle.dump({'classes': classes, 'val_scores': val_scores, 'val_labels': val_labels}, fw)

    print("Computing kernel")
    K = intersection_kernel(test_bows, train_bows)

    print("Scoring test data")
    scores = []
    for m in tqdm(models):
        results, binary_scores, probs = test_binary(K, m)
        scores.append(probs)  # Use probabilities -> better multiclass accuracy
    
    labels = np.argmax(np.array(scores), 0)

    # Map topics to labels
    classes2labels = {'society': 0, 'economy': 1, 'science': 8, 'Спорт': 4, 'Бывший СССР': 3, 'Силовые структуры': 2, 'Туризм': 7, 'Строительство': 6, 'Забота о себе': 5, 'Россия': 0}
    n_classes = len(set(classes2labels.values()))
    unknown_labels = list(set(range(n_classes)) - set(classes2labels.values()))
    if len(unknown_labels) > 0:
        final_labels = [classes2labels.get(classes[y], unknown_labels[randint(0, len(unknown_labels)-1)]) for y in labels]
    else:
        final_labels = [classes2labels[classes[y]] for y in labels]

    # Reorder/merge scores
    new_scores = -10000.0 * np.ones((n_classes, len(scores[0])))
    for i, y in enumerate(classes):
        new_scores[classes2labels[y]] = np.maximum(new_scores[classes2labels[y]], scores[i])
    with open("test_scores_new22.pickle", "wb") as fw:
        pickle.dump(np.array(new_scores), fw)

    print("Saving results")
    with open("submission22.csv", 'w') as fw:
        fw.write("topic,index\n")
        for i, y in enumerate(final_labels):
            fw.write(f"{y},{i}\n")