My code for the [news scraping / topic prediction](https://www.kaggle.com/competitions/news-scraping-competition) kaggle competition.

You can find the scraping code (better than mine) in the competition Code section.

# Before running train\_kernel\_svm.py build the Cython extension
    python setup.py build_ext --inplace

# Contents:

* train\_kernel\_svm.py - approach #1
* train\_rubert.py - approach #2 (trained on different data)
* fusion.py - late fusion (gives +0.5% to using approach #2)
* fix\_known\_documents.py - set labels of the test documents that appear in the training set (4Gb RAM)

Tested on MacOS / kaggle
