def K_ano_Titan(k=8):
    import pandas as pd
    import numpy as np

    THE_MOST_IMPORTANT_K = k  # Changed this, so i can use it in the split function as well, without any kind of refactoring.

    df = pd.read_csv("./titanic/test.csv", sep=",", header=0, index_col=False,
                     engine='python');  # We load the data using Pandas

    df.drop(columns=["PassengerId", "Name"], inplace=True)  # dropped because unique for every row
    df.drop(columns=["Ticket", "Cabin"], inplace=True)  # dropped because almost unique for every row
    df.dropna(inplace=True)

    categorical = [
        'Survived',
        'Pclass',
        'Sex',
        'SibSp',
        'Parch',
        'Embarked'
    ]

    for name in categorical:
        df[name] = df[name].astype('category')

    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    df = df[cols]

    def get_spans(df, partition, scale=None):
        """
        :param        df: the dataframe for which to calculate the spans
        :param partition: the partition for which to calculate the spans
        :param     scale: if given, the spans of each column will be divided
                          by the value in `scale` for that column
        :        returns: The spans of all columns in the partition
        """
        spans = {}
        for column in df.columns:
            if column in categorical:
                span = len(df[column][partition].unique())
            else:
                span = df[column][partition].max() - df[column][partition].min()
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans

    full_spans = get_spans(df, df.index)
    full_spans

    def split(df, partition, column):
        """
        :param        df: The dataframe to split
        :param partition: The partition to split
        :param    column: The column along which to split
        :        returns: A tuple containing a split of the original partition
        """
        dfp = df[column][partition]
        if column in categorical:
            # Commenting out the next 4 lines should make the world a better place, but who am i to know this for sure...
            # But it does not... Reality is
            values = dfp.unique()
            lv = set(values[:len(values) // 2])
            rv = set(values[len(values) // 2:])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]

            indexes = dfp.value_counts().index
            values = dfp.value_counts().values

            zipped = zip(indexes, values)
            sortedvals = np.array(sorted(zipped, key=lambda x: x[1]))
            sortedvalues = np.array(sortedvals[:, 1], dtype=int)
            cumulated = np.cumsum(sortedvalues)

            bestindex = 0
            for i in range(len(cumulated)):
                if cumulated[i] >= THE_MOST_IMPORTANT_K and (cumulated[-1] - cumulated[i]) >= THE_MOST_IMPORTANT_K and \
                                cumulated[i] < cumulated[-1] / 2:
                    bestindex = i
            if cumulated[bestindex + 1] >= THE_MOST_IMPORTANT_K and (
                cumulated[-1] - cumulated[bestindex + 1]) >= THE_MOST_IMPORTANT_K:
                if np.abs(cumulated[-1] / 2 - cumulated[bestindex]) > np.abs(cumulated[-1] / 2 - cumulated[bestindex + 1]):
                    bestindex += 1

            lv = set(sortedvals[:bestindex, 0])
            rv = set(sortedvals[bestindex:, 0])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]

            if len(dfl) < THE_MOST_IMPORTANT_K or len(dfr) < THE_MOST_IMPORTANT_K:
                median = dfp.mean()  # Very clever hack, 200IQ code... xD

            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]

            return (dfl, dfr)

    def is_k_anonymous(df, partition, sensitive_column):
        """
        :param               df: The dataframe on which to check the partition.
        :param        partition: The partition of the dataframe to check.
        :param sensitive_column: The name of the sensitive column
        :param                k: The desired k
        :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
        """
        # COMMENT OUT THE NEXT 4 LINES IF WE DO NOT WANT TO ENFORCE DIFFERENT SENSITIVE COLUMNS IN PARTITIONS

        # sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        # for sensitive_value, count in sensitive_counts[sensitive_column].items():
        # if count == 0:
        # return False

        if len(partition) < THE_MOST_IMPORTANT_K:
            return False
        return True


    def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
        """
        :param               df: The dataframe to be partitioned.
        :param  feature_columns: A list of column names along which to partition the dataset.
        :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
        :param            scale: The column spans as generated before.
        :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
        :returns               : A list of valid partitions that cover the entire dataframe.
        """
        finished_partitions = []
        partitions = [df.index]
        while partitions:
            partition = partitions.pop(0)
            # Lil' faster
            if len(partition) < 2 * THE_MOST_IMPORTANT_K:
                finished_partitions.append(partition)
                continue
            spans = get_spans(df[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = split(df, partition, column)
                if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions

    feature_columns = ['Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    sensitive_column = 'Survived'
    finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)

    def agg_categorical_column(series):
        series = series.astype(str)
        return [','.join(set(series))]


    def agg_numerical_column(series):
        return [series.mean()]

    def build_anonymized_dataset_2(df, partitions, feature_columns, sensitive_column, max_partitions=None):
        aggregations = {}
        for column in feature_columns:
            if column in categorical:
                aggregations[column] = agg_categorical_column
            else:
                aggregations[column] = agg_numerical_column
        rows = []
        # print(aggregations)
        # print(partitions)
        for i, partition in enumerate(partitions):
            if i % 10 == 1:
                print("Finished {} partitions...".format(i))
            if max_partitions is not None and i > max_partitions:
                break
            grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
            sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column: 'count'})
            values = grouped_columns.iloc[0].to_dict()
            IsOkay = True
            # COMMENT OUT THE NEXT 3 LINES IF WE DO NOT WANT TO ENFORCE DIFFERENT SENSITIVE COLUMNS IN PARTITIONS
            # for sensitive_value, count in sensitive_counts[sensitive_column].items():
            # if count == 0:
            # IsOkay=False
            if IsOkay:
                for rowind in range(len(df.loc[partition])):
                    currow = df.loc[partition].iloc[rowind, :].copy()
                    for feature in feature_columns:
                        currow[feature] = grouped_columns[feature][0]
                    rows.append(currow.copy())
        return pd.DataFrame(rows)

    dfn2 = build_anonymized_dataset_2(df, finished_partitions, feature_columns, sensitive_column)

    import matplotlib.pyplot as plt
    import numpy as np

    sizes = np.zeros(len(finished_partitions))
    for i in range(len(finished_partitions)):
        sizes[i] = finished_partitions[i].shape[0]

    plt.figure()
    plt.hist(sizes)
    plt.savefig(str(THE_MOST_IMPORTANT_K) + "_titanic.png")

    uniqitems = {}
    for col in categorical:
        uniqitems[col] = df[col].unique()

    rows = []

    for rowind in range(len(dfn2)):
        if rowind % 71 == 0:
            print(rowind / len(dfn2) * 100)
        currow = dfn2.iloc[rowind, :].copy()
        for col in categorical:
            if col != 'Survived':
                values = str(currow[col]).split(',')

                for possibleitem in uniqitems[col]:
                    possibleitem = str(possibleitem)
                    if possibleitem in values:
                        currow = currow.append(pd.Series([1 / len(values)], [col + '_' + possibleitem]))
                    else:
                        currow = currow.append(pd.Series([0], [col + '_' + possibleitem]))
                currow = currow.drop(col)
        rows.append(currow.copy())

    final_set = pd.DataFrame(rows)

    final_set.to_csv(str(THE_MOST_IMPORTANT_K) + "_titanic.csv")

    return final_set