import numpy as np
import pandas as pd
import re


class SimilarityScores:
    """ Hold scores for determining which network best approximates community.
    """

    def __init__(self, num_communities, num_networks=20):
        """ Initialize places to hold scores for communities.
        """

        self.num_communities = num_communities
        self.num_networks = num_networks

        self.community = np.zeros(num_communities)
        self.func_conn = np.zeros(num_communities)
        self.spatial_score = np.zeros(num_communities)
        self.confidence = np.zeros(num_communities)
        self.r = np.zeros(num_communities)
        self.g = np.zeros(num_communities)
        self.b = np.zeros(num_communities)
        self.network = dict()
        self.network_manual_decision = dict()

        self.alt_networks = dict()
        self.alt_func_sims = dict()
        self.alt_spatial_scores = dict()
        for i in range(num_networks - 1):
            self.alt_networks[i] = dict()
            self.alt_func_sims[i] = dict()
            self.alt_spatial_scores[i] = dict()

    def __str__(self):
        return f"Scores for {self.num_communities} communities"

    def report(self):
        print(f"Community: {self.community.astype(int)}")
        print("Network: [{}]".format(
            ', '.join([f"{k}: {v}" for k, v in self.network.items()])
        ))
        print("Network Manual Decision: [{}]".format(
            ', '.join([f"{k}: {v}" for k, v in self.network_manual_decision.items()])
        ))
        for desc, val in [
            ("R", self.r), ("G", self.g), ("B", self.b),
            ("FC_Similarity", self.func_conn),
            ("Spatial_Score", self.spatial_score), ("Confidence", self.confidence),
        ]:
            print("{}: [{}]".format(
                desc, ','.join([f'{_:0.4f}' for _ in val])
            ))

    def to_dataframe(self):
        """ Structure data as a pandas dataframe. """

        # The data structures, lists and dicts willy-nilly without meaningful keys
        # are super-shitty and frustrating. TODO: Clean these up.
        if len(self.network_manual_decision) == 0:
            _network_manual_decision = list('' for _ in range(self.num_communities))
        else:
            _network_manual_decision = [v for k,v in self.network_manual_decision.items()]
        d = {
            "Community": self.community,
            "Network": [v for k,v in self.network.items()],
            "Network_Manual_Decision": _network_manual_decision,
            "R": self.r,
            "G": self.g,
            "B": self.b,
            "FC_Similarity": self.func_conn,
            "Spatial_Score": self.spatial_score,
            "Confidence": self.confidence,
        }
        for i in range(self.num_networks - 1):
            d[f"Alt_Network_{i + 1:02d}"] = [v for k,v in self.alt_networks[i].items()]
            d[f"Alt_FC_Similarity_{i + 1:02d}"] = [v for k,v in self.alt_func_sims[i].items()]
            d[f"Alt_Spatial_Score_{i + 1:02d}"] = [v for k,v in self.alt_spatial_scores[i].items()]

        # for k, v in d.items():
        #     print(f"{k}: {len(v)}")
        return pd.DataFrame.from_dict(d, orient='columns')

    def to_excel(self, file_path):
        """ Save data to an Excel spreadsheet. """

        # In Lynch's matlab, "writetable(struct2table(S), file_path);"
        self.to_dataframe().to_excel(file_path, index=False)

    def from_excel(self, file_path):
        """ Load data from an Excel spreadsheet. """

        _df = pd.read_excel(file_path)

        alt_pattern = re.compile("(Alt_.+)_([0-9]+)")
        for col in _df.columns:
            match_alt = alt_pattern.match(col)
            if match_alt:
                if match_alt.group(1) == "Alt_Network":
                    if not hasattr(self, "alt_networks"):
                        setattr(self, "alt_networks", dict())
                    self.alt_networks[int(match_alt.group(2))] = dict()
                    for i, val in enumerate(_df[col].values):
                        self.alt_networks[int(match_alt.group(2))][(i, 0)] = val
                elif match_alt.group(1) == "FC_Similarity":
                    if not hasattr(self, "alt_func_sims"):
                        setattr(self, "alt_networks", dict())
                    self.alt_func_sims[int(match_alt.group(2))] = dict()
                    for i, val in enumerate(_df[col].values):
                        self.alt_func_sims[int(match_alt.group(2))][(i, 0)] = val
                elif match_alt.group(1) == "Spatial_Score":
                    if not hasattr(self, "alt_spatial_scores"):
                        setattr(self, "alt_networks", dict())
                    self.alt_spatial_scores[int(match_alt.group(2))] = dict()
                    for i, val in enumerate(_df[col].values):
                        self.alt_spatial_scores[int(match_alt.group(2))][(i, 0)] = val
            else:
                if col == "Community":
                    self.community = _df[col].values
                elif col == "Network":
                    self.network = _df[col].values
                elif col == "Network_Manual_Decision":
                    self.network_manual_decision = _df[col].values
                elif col == "R":
                    self.r = _df[col].values
                elif col == "G":
                    self.g = _df[col].values
                elif col == "B":
                    self.b = _df[col].values
                elif col == "FC_Similarity":
                    self.func_conn = _df[col].values
                elif col == "Spatial_Score":
                    self.spatial_score = _df[col].values
                elif col == "Confidence":
                    self.confidence = _df[col].values

