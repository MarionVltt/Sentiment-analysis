import numpy as np
import pandas as pd
import re
# import spacy
# import test
# from custom_multiprocessing import Pandas_mpx

pd.options.mode.chained_assignment = None  # default='warn'



def unwrap_SingleRaw_self_get_postag_raw(*arg, **kwarg):
    """
    Workaround: utility function to unwrap self to before multiprocessing
    for SingleRaw.get_postag_raw class method
    """
    return SingleRaw.get_postag_raw(*arg, **kwarg)


class SingleRaw():
    """
    Class to clean one single raw
    - self.tokens: all tokens extracted from the string
    - self.clean_text: string without punctuation, spaces (lowercase option), non alpha signs

    TODO:

    """

    STOPWORDS_FR = ["bonjour", "salut", "slt", "au revoir", "a bientôt", "a bientot"]

    def __init__(self, sentence):
        self.type = type(sentence)
        self.text = str(sentence)
        self.clean_text = None
        self.default_case = False

    def __repr__(self):
        return "SingleRaw: text({}), " \
              "clean_text({}), ".format(self.text, self.clean_text)
        #return "SingleRaw: text({}), " \
        #       "clean_text({}), " \
        #      "tokens({}), ".format(self.text, self.clean_text, self.tokens)

    def standardize_raw(self, sequence_to_remove=[]):
        """
        Initialize self.clean_text:
        removes punctuation and special characters (except periods and exclamation marks), multiple spaces,
        urls, mention with @, numbers, the stopwords present in STOPWORDS_FR and the regex in sequence_to_remove
        from raw text and stores self.clean_text
        example seq to remove : "<br\s*/><br\s*/>" for IMDb
        """
        no_tabs = self.text.lower().replace('\t', ' ')
        remove_tag = re.sub(r'@[A-Za-z0-9]+', "", no_tabs)
        remove_long_url = re.sub(r'https?://[A-Za-z0-9./]+', "", remove_tag)
        remove_short_url = re.sub(r'www\.?[A-Za-z0-9./]+', "", remove_long_url)
        seq_removed = remove_short_url
        for seq in sequence_to_remove:
            seq_removed = re.sub(seq, " ", seq_removed)
        alpha_only = re.sub("[^a-zA-Zàáâãäåçèéêëìíîïðòóôõöùúûüýÿ\!\.]", " ", seq_removed)
        no_stop = alpha_only
        for s in SingleRaw.STOPWORDS_FR:
            if s in alpha_only:
                no_stop = re.sub(s, "", alpha_only)
        self.clean_text = re.sub(" +", " ", no_stop)

        if len(self.clean_text.split()) == 0:
            self.default_case = True

        return self

    # def is_default(self, sequence_to_remove=[]):
    #     """
    #     Checks whether it is possible to predict the class:
    #     No : Default case = True
    #     Yes: Default case = False
    #     - Standardizes raw
    #     Tests number of remaining words to compute default case
    #
    #     """
    #
    #     self.standardize_raw(sequence_to_remove=sequence_to_remove)
    #
    #     # default case: if there only remain less than 2 words in the pipe
    #     if len(self.clean_text.split()) < 2:
    #         self.default_case = True
    #
    #     return self


class RangeRaw():
    """
    Class to clean a range of raw
    input:
      file_path (csv file link one single columns)
      index_transaction: column index text location
    TODO:
       apply multiprocessing when needed
    """
    multiprocessing = False

    @classmethod
    def setMultiprocessing(cls, multiprocessing):
        cls.multiprocessing = multiprocessing

    def __init__(self, dataframe, index_sentences, index_labels):

        #self.df = pd.read_csv(str(file_path), sep=';', header=None, index_col=0)
        self.df = dataframe
        #print(self.df.head())
        self.col_names = self.df.columns.tolist()
        self.col_name_sentences = self.col_names[index_sentences]
        self.col_name_labels = self.col_names[index_labels]
        self.df.dropna(inplace=True)

        if RangeRaw.multiprocessing:
            self.df_class = Pandas_mpx.map(SingleRaw,
                                           self.df[self.col_name_sentences])
        else:
            self.df_class = self.df[self.col_name_sentences].apply(SingleRaw)

    def standardize_range(self, sequence_to_remove=[]):
        """
        Calls standardize_raw() for each row of the Dataframe
        creates column clean_text on self.df
        updates self.col_names accordingly
        """
        # print(self.df_class.head())
        if RangeRaw.multiprocessing:
            self.df_class = Pandas_mpx.map(unwrap_SingleRaw_self_get_postag_raw,
                                           self.df_class)
        else:
            self.df_class.apply(lambda singleRaw: singleRaw.standardize_raw(sequence_to_remove=sequence_to_remove))
        # print(self.df_class.head())
        self.df[['clean_text']] = pd.DataFrame(
            self.df_class.apply(lambda x: [x.clean_text]).values.tolist(), index=self.df.index)
        self.col_names = self.df.columns.tolist()
        # print(self.df.clean_text.head())
