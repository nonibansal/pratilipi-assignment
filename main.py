import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class Recommendation:
    def __init__(self, content_data, data):
        self.logger = logging.getLogger(__name__)
        data = data.loc[:, ["user_id", "pratilipi_id", "read_percent", "updated_at"]]
        data = data.rename(columns={"updated_at": "date_read"})
        data = (data.assign(date_read=lambda x: pd.to_datetime(x["date_read"]))
                .sort_values(by=["date_read"])
                .reset_index(drop=True))

        mark_75 = data.shape[0] * 0.75
        self.train = data.loc[0:mark_75 - 1].reset_index(drop=True)
        self.test = data.loc[mark_75:].reset_index(drop=True)
        self.content_data = content_data
        self.user_data_recommendation = None
        self.content_data_recommendation = None
        del data
        self.preprocess_user_data()
        self.preprocess_content_data()

    @staticmethod
    def normalize(pred_ratings):
        return (pred_ratings - pred_ratings.min()) / (pred_ratings.max() - pred_ratings.min())

    @staticmethod
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @staticmethod
    def one_hot_encode(df, enc_col):
        ohe_df = pd.get_dummies(df[enc_col])
        ohe_df.reset_index(drop=True, inplace=True)
        return pd.concat([df, ohe_df], axis=1)

    def preprocess_user_data(self):
        train = self.train.groupby(["user_id", "pratilipi_id"]).agg({"read_percent": "mean"}).reset_index()
        interesting_users = train.groupby("user_id")["pratilipi_id"].count().reset_index(name="count")
        interesting_users = interesting_users.loc[interesting_users["count"] >= 20, "user_id"].tolist()
        interesting_pratilipi = train.groupby("pratilipi_id")["user_id"].count().reset_index(name="count")
        interesting_pratilipi = interesting_pratilipi.loc[interesting_pratilipi["count"] >= 20,
                                                          "pratilipi_id"].tolist()
        train = train.loc[(train["user_id"].isin(interesting_users)) &
                          (train["pratilipi_id"].isin(interesting_pratilipi))].reset_index(drop=True)

        pratilipi_ids = list(train["pratilipi_id"].unique())
        user_ids = list(train["user_id"].unique())
        user_ids_dict = {}
        pratilipi_ids_dict = {}
        k = 0
        for user_id in user_ids:
            user_ids_dict[user_id] = k
            k += 1
        k = 0
        for pratilipi_id in pratilipi_ids:
            pratilipi_ids_dict[pratilipi_id] = k
            k += 1

        data_matrix = np.zeros((len(user_ids), len(pratilipi_ids))).astype(np.float16)
        for i in range(train.shape[0]):
            user_id_ix = user_ids_dict.get(train.loc[i, "user_id"])
            pratilipi_id_ix = pratilipi_ids_dict.get(train.loc[i, "pratilipi_id"])
            read_percent = train.loc[i, "read_percent"]

            data_matrix[user_id_ix, pratilipi_id_ix] = read_percent
        del train

        data_matrix = csr_matrix(data_matrix)
        u, s, v = svds(data_matrix, k=100)
        s = np.diag(s)
        pred_ratings = np.dot(np.dot(u, s), v)
        pred_ratings = self.normalize(pred_ratings)
        pred_ratings = pd.DataFrame(
            pred_ratings,
            columns=pratilipi_ids,
            index=user_ids
        ).transpose()
        del u
        del s
        del v
        del data_matrix
        self.user_data_recommendation = pred_ratings

    def preprocess_content_data(self):

        self.content_data["category_name_present"] = 0
        self.content_data.loc[self.content_data["category_name"].isna() is False, "category_name_present"] = 1

        self.content_data = (pd.pivot(self.content_data, index=["author_id", "pratilipi_id", "reading_time",
                                                                "updated_at", "published_at"],
                                      columns="category_name", values="category_name_present")
                             .reset_index().fillna(0).rename_axis(None, axis=1))
        self.content_data = (self.content_data.assign(published_at=lambda x: pd.to_datetime(x["published_at"]))
                             .assign(published_year=lambda x: pd.to_datetime(x["published_at"]).dt.year)
                             .assign(author_id=lambda x: x["author_id"].astype(str).fillna("author_nan"))
                             .assign(
            published_year=lambda x: x["published_year"].astype(str).fillna("published_year_nan")))

        self.content_data = self.content_data.drop(columns=["published_at", "updated_at"])

        label_encoder_author = LabelEncoder()
        label_encoder_author.fit(self.content_data["author_id"])
        self.content_data = self.content_data.assign(author_id=lambda x: label_encoder_author.transform(x["author_id"]))
        ohe_df = pd.get_dummies(self.content_data["published_year"])
        ohe_df.reset_index(drop=True, inplace=True)
        self.content_data = pd.concat([self.content_data, ohe_df], axis=1)

        # Features Extracted from User Data
        pratilipi_reads = self.train.groupby("pratilipi_id").size().reset_index(name="total_reads")
        pratilipi_percent_read = (self.train.groupby("pratilipi_id")
                                  .agg({"read_percent": "mean"})
                                  .reset_index())
        pratilipi_unique_reads = (self.train.groupby("pratilipi_id")
                                  .agg({"user_id": "nunique"})
                                  .reset_index()
                                  .rename(columns={"user_id": "unique_reads"}))
        pratilipi_50_unique_reads = (self.train.loc[self.train["read_percent"] >= 50.0, :]
                                     .reset_index(drop=True))
        pratilipi_50_unique_reads = (pratilipi_50_unique_reads.groupby("pratilipi_id")
                                     .agg({"user_id": "nunique"})
                                     .reset_index()
                                     .rename(columns={"user_id": "unique_50_reads"}))
        pratilipi_50_reads = (self.train.loc[self.train["read_percent"] >= 50.0, :]
                              .reset_index(drop=True))
        pratilipi_50_reads = (pratilipi_50_reads.groupby("pratilipi_id")
                              .size()
                              .reset_index(name="total_50_reads"))
        self.content_data = (self.content_data.merge(pratilipi_reads, on="pratilipi_id", how="left")
                             .fillna(0))
        self.content_data = (self.content_data.merge(pratilipi_percent_read, on="pratilipi_id", how="left")
                             .fillna(0))
        self.content_data = (self.content_data.merge(pratilipi_unique_reads, on="pratilipi_id", how="left")
                             .fillna(0))
        self.content_data = (self.content_data.merge(pratilipi_50_unique_reads, on="pratilipi_id", how="left")
                             .fillna(0))
        self.content_data = (self.content_data.merge(pratilipi_50_reads, on="pratilipi_id", how="left")
                             .fillna(0))
        self.content_data = self.content_data.drop(["published_year"], axis=1)

        min_max_columns = ["reading_time", "total_reads", "read_percent", "unique_reads",
                           "unique_50_reads", "total_50_reads"]
        min_max_scale = MinMaxScaler()
        min_max_scale.fit(self.content_data[min_max_columns])
        self.content_data[min_max_columns] = min_max_scale.transform(self.content_data[min_max_columns])

        self.content_data_recommendation = self.content_data.set_index("pratilipi_id")

    def get_top_n_pratilipi_from_content(self, pratilipi_id, n_recs=5):
        try:
            inputVec = self.content_data_recommendation.loc[pratilipi_id].values
            self.content_data_recommendation["sim"] = self.content_data_recommendation.apply(
                lambda x: self.cosine_sim(inputVec, x.values), axis=1)
            output = self.content_data_recommendation.nlargest(columns="sim", n=n_recs)
            return output.reset_index().loc[:, ["pratilipi_id", "sim"]]
        except Exception as e:
            self.logger.exception(e)
            return pd.DataFrame(columns=["pratilipi_id", "sim"])

    def get_top_n_pratilipi_from_userid(self, user_id, n_recs=5):
        try:
            usr_pred = (self.user_data_recommendation[user_id]
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={user_id: "sim"}))
            usr_pred = (usr_pred.sort_values(by="sim", ascending=False).head(n_recs)
                        .rename(columns={"index": "pratilipi_id"}))
            return usr_pred
        except Exception as e:
            self.logger.exception(e)
            return pd.DataFrame(columns=["pratilipi_id", "sim"])

    def predict(self):
        output = {}
        test_userids = list(self.test["user_id"].unique())
        for user_id in test_userids:
            dummy = self.get_top_n_pratilipi_from_userid(user_id, 5)
            already_read = []
            for pratilipi_id in list(self.train.loc[self.test["user_id"] == user_id, "pratilipi_id"].unique()):
                dummy = pd.concat([dummy, self.get_top_n_pratilipi_from_content(pratilipi_id, 5)])
                already_read.append(pratilipi_id)
            dummy = dummy.sort_values(by="sim", ascending=False).reset_index(drop=True)
            dummy = dummy.loc[dummy["pratilipi_id"].notin(already_read)].reset_index(drop=True)
            output[user_id] = dummy.loc[0:4, "pratilipi_id"].tolist()
        return output
