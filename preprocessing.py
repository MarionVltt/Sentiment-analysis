# usr/bin/python3

import re
import pandas as pd


class SingleRaw():
    """
    Class to clean a single sentence

    Attributes:
        - self.text: initial sentence
        - self.clean_text: string without punctuation, spaces (lowercase option), non alpha signs
        - self.default_case : True when clean_text is empty, no prediction possible
        - self.language : English or French
        - self.stopwords : common words which need to be removed

    Methods:
        - standardize_raw : clean the initial sentence

    """

    def __init__(self, sentence, language='English'):
        self.text = str(sentence)
        self.clean_text = None
        self.default_case = False
        self.language = language
        if language == 'English':
            self.stopwords = ["hello", "hi", "hey",
                              "good morning", "bye", "see you"]
        elif language == 'French':
            self.stopwords = ["bonjour", "salut", "slt",
                              "au revoir", "a bientôt", "a bientot"]
        else:
            print("Please specify correct language (English or French)")

    def __repr__(self):
        return "SingleRaw: text({}), " \
            "clean_text({}), ".format(self.text, self.clean_text)

    def standardize_raw(self, sequence_to_remove=[]):
        """
        Initialize self.clean_text
        removes punctuation and special characters (except periods and exclamation marks), multiple spaces,
        urls, mention with @, numbers, the stopwords present in self.stopwords and the regex in sequence_to_remove
        from raw text and stores self.clean_text
        example of sequence to remove : "<br\s*/><br\s*/>" for IMDb

        Parameters:
            - sequence_to_remove: particular string to remove from the sentence
        """
        no_tabs = self.text.lower().replace('\t', ' ')
        remove_tag = re.sub(r'@[A-Za-z0-9]+', "", no_tabs)
        remove_long_url = re.sub(r'https?://[A-Za-z0-9./]+', "", remove_tag)
        remove_short_url = re.sub(r'www\.?[A-Za-z0-9./]+', "", remove_long_url)
        seq_removed = remove_short_url
        for seq in sequence_to_remove:
            seq_removed = re.sub(seq, " ", seq_removed)
        if self.language == 'English':
            alpha_only = re.sub("[^a-zA-Z\!\.]", " ", seq_removed)
        elif self.language == 'French':
            alpha_only = re.sub(
                "[^a-zA-Zàáâãäåçèéêëìíîïðòóôõöùúûüýÿ\!\.]", " ", seq_removed)

        no_stop = alpha_only
        for s in self.stopwords:
            if s in alpha_only:
                no_stop = re.sub(s, "", alpha_only)
        self.clean_text = re.sub(" +", " ", no_stop)

        if len(self.clean_text.split()) == 0:
            self.default_case = True

        return self


class RangeRaw():
    """
    Class to clean a range of raw sentences (used only to preprocess the sentences before training a model)
    Input:
      - dataframe (pd.DataFrame with at least a column with the sentences and one with the corresponding labels (positive or negative))
      - index_sentences (int): index of the column containing the text
      - index_labels (int):  index of the column containing the labels

     Attributes:
        - self.df: the dataframe with the sentences and labels
        - self.col_names: list of all columns names
        - self.col_name_sentences: name of the column containing the text
        - self.col_name_labels: name of the column containing the labels
        - self.df_class: sentences casted as SingleRaw

    Methods:
        - standardize_range : clean all the sentences
    """

    def __init__(self, dataframe, index_sentences, index_labels, language='English'):

        self.df = dataframe
        self.col_names = self.df.columns.tolist()
        self.col_name_sentences = self.col_names[index_sentences]
        self.col_name_labels = self.col_names[index_labels]
        self.df.dropna(inplace=True)
        self.df_class = self.df[self.col_name_sentences].apply(
            lambda x: SingleRaw(x, language))

    def standardize_range(self, sequence_to_remove=[]):
        """
        Calls standardize_raw() for each row of the Dataframe
        Creates column clean_text on self.df
        Updates self.col_names accordingly

        Parameters:
            - sequence_to_remove: particular string to remove from the sentences
        """

        self.df_class.apply(lambda x: x.standardize_raw(
            sequence_to_remove=sequence_to_remove))
        self.df[['clean_text']] = pd.DataFrame(
            self.df_class.apply(lambda x: [x.clean_text]).values.tolist(), index=self.df.index)
        self.col_names = self.df.columns.tolist()
