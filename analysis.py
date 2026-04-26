import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

pio.renderers.default = "browser"

# reading
votes = pd.read_csv('data/HS119_votes.csv')
members = pd.read_csv('data/HS119_members.csv')
roll_calls = pd.read_csv('data/HS119_rollcalls.csv')

house_votes = votes[votes["chamber"] == "House"]
senate_votes = votes[votes["chamber"] == "Senate"]

house_members = members[members["chamber"] == "House"]
senate_members = members[members["chamber"] == "Senate"]

# mapping
def convert_vote(code):
    if code in [1, 2, 3]:
        return 1;
    elif code in [4, 5, 6]:
        return -1;
    else:
        return 0;

house_votes["votes_numeric"] = votes["cast_code"].apply(convert_vote)
senate_votes["votes_numeric"] = votes["cast_code"].apply(convert_vote)

house_matrix = house_votes.pivot_table(index="icpsr", columns="rollnumber", values="votes_numeric", fill_value=0)
senate_matrix = senate_votes.pivot_table(index="icpsr", columns="rollnumber", values="votes_numeric", fill_value=0)

# Keep only legislators with at least 100 cast votes (non-zero entries).
house_matrix = house_matrix[(house_matrix != 0).sum(axis=1) >= 100]
senate_matrix = senate_matrix[(senate_matrix != 0).sum(axis=1) >= 100]


# L2-normalize each legislator vector (row) before PCA
house_matrix_norm = pd.DataFrame(
    normalize(house_matrix, norm="l2", axis=1),
    index=house_matrix.index,
    columns=house_matrix.columns,
 )
senate_matrix_norm = pd.DataFrame(
    normalize(senate_matrix, norm="l2", axis=1),
    index=senate_matrix.index,
    columns=senate_matrix.columns,
 )

# PCA

pca = PCA(n_components=2)
coords_house = pca.fit_transform(house_matrix_norm)
coords_senate = pca.fit_transform(senate_matrix_norm)

# plot house
df_house = pd.DataFrame(coords_house, columns=["PC1", "PC2"])
df_house["icpsr"] = house_matrix_norm.index

df_house = df_house.merge(
    house_members[["icpsr", "party_code", "bioname"]],
    on="icpsr"
)


fig = px.scatter(
    df_house,
    title="House",
    x="PC1",
    y="PC2",
    color="party_code",
    hover_name="bioname"
)

fig.show()

# plot senate
df_senate = pd.DataFrame(coords_senate, columns=["PC1", "PC2"])
df_senate["icpsr"] = senate_matrix_norm.index

df_senate = df_senate.merge(
    senate_members[["icpsr", "party_code", "bioname"]],
    on="icpsr"
)


fig = px.scatter(
    df_senate,
    title="Senate",
    x="PC1",
    y="PC2",
    color="party_code",
    hover_name="bioname"
)

fig.show()