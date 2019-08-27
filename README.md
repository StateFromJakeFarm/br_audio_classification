# Audio Classification Using Ensemble of Binary Relevance Classifiers
---

## Ensemble of What Now?
If these terms are as unfamiliar to you as they were to me at first, I've written a [blog post](https://jakeahansen.com/blog/view/audio-classification-using-ensemble-of-binary-relevance-classifiers) to give an overview of the purpose and methodology behind this project.

## Scraper & Classifier
This repository contains two separate programs: a web scraper to collect data, and a classifier to classify that data.

### Scraper
The scraper attempts to locate a search bar within every user-provided webpage, and if successful, it will search each website for all user-specified search terms.  The scraper will then scrape any files whose titles contain more than a certain percentage of the search terms.

As an example, say we have the following configuration:
```
start_urls       = 'https://freesound.org'
search_terms     = 'car', 'horn'
accept_threshold = 10%
max_page         = 3
```

After locating the search bar on the landing page, the scraper will crawl the following two URLs: `https://freesound.org/search/?q=car` and `https://freesound.org/search/?q=horn`.

On the first of these two pages, it will find a WAV file titled "Car Alarm Horn, Walking Past, A.wav".  It will split this title up to yield: `['car', 'alarm', 'horn,', 'walking', 'past,', 'A', 'wav']`.  Since 2 out of the 7 terms within this title come from our search terms, the percentage of search terms (29%) is greater than our minimum acceptance threshold, and so the file will be scraped.

In each page it inspects, the scraper will also look for "next page" links.  It will follow these links up until the `max_page` index, combing each page for files to scrape.

The scraper is configured to use a GCP Storage Bucket to house scraped files, though it can be configured to use any storage backend supported by [Scrapy](https://scrapy.org/).

#### Options
Once Scrapy is installed, the spider is run by navigating to the `br_audio_classification/soundScrape/` directory and entering the following command:

`$ scrapy crawl sound [args]`

**Note:** Arguments are passed to the scraper using the `-a` flag followed by `<key>=<value>`.  See [this page](https://docs.scrapy.org/en/latest/topics/spiders.html#spider-arguments) for more info.

Arguments for the scraper are as follows:

* `start_urls`: (string) comma-separated list of webpages to be scraped
* `search_terms`: (string) comma-separated search terms
* `avoid_terms`: (string) comma-separated terms to avoid (when found in file's title, they subtract from its match percentage)
* `accept_threshold`: (integer) minimum percentage of matched search terms within a file's title for that file to be scraped
* `max_page`: (integer) highest "next page" link to visit

Example:
```
$ scrapy crawl sound -a start_urls=https://freesound.org,http://soundbible.com -a search_terms=car,horn -a avoid_terms=french,brass -a accept_threshold=10 -a max_page=3
```

Scraped files are named with a list of all matched search terms, followed by a hash of the file's URI.  For example: `car-horn_0847bda76017249078b72a3571653ac106d28f92.mp3`.

### Classifier
Because the [blog post](https://jakeahansen.com/blog/view/audio-classification-using-ensemble-of-binary-relevance-classifiers) discusses the system's purpose and design in-depth, this document will focus on the program's features and usage.

#### Datasets
The classifier currently supports the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), as well as datasets gathered by the scraper.  Use the `--path` argument to specify the absolute or relative path to the dataset.  If using a dataset gathered by the scraper, the target search terms must also be specified using the `--gathered` argument.  To see a full list of all arguments, run the following command:

```
$ python3 classifier.py -h
```

#### Example Training Commands
```
$ python3 classifier.py -p ./UrbanSound8K/ --hidden 256 -d 4 -r 3 -e 100 --dropout 0.3 --sr 44100 -s 256h_3d_3r_30dropout_20e_urbansound8k

$ python classifier.py -p ./soundSortAudio/ --hidden 256 -d 3 -r 3 -e 20 --dropout 0.3 --sr 16000 -g horn,children,bark,drill,engin,gun,siren,saw -s 256h_3d_3r_30dropout_20e_soundScrape
```

#### Example Testing Command
```
$ python3 classifier.py -p ./UrbanSound8K/ -l ./saved_models/256h_3d_3r_30dropout_20e_urbansound8k
```