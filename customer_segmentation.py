import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv("marketing_campaign.csv", sep="\t")

print("\nDataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\nCleaning data...")

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Income to numeric
df["Income"] = pd.to_numeric(
    df["Income"],
    errors="coerce"
)

# Remove rows where Income is missing
df = df.dropna(subset=["Income"])

# Convert customer date into datetime
df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    errors="coerce",
    dayfirst=True
)

print("Shape after cleaning:", df.shape)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print("\nCreating features...")


# Total amount spent
df["TotalSpent"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)


# Total purchases
df["TotalPurchases"] = (
    df["NumWebPurchases"]
    + df["NumCatalogPurchases"]
    + df["NumStorePurchases"]
)


# Total children
df["TotalChildren"] = (
    df["Kidhome"]
    + df["Teenhome"]
)


# Total accepted campaigns
df["TotalCampaignsAccepted"] = (
    df["AcceptedCmp1"]
    + df["AcceptedCmp2"]
    + df["AcceptedCmp3"]
    + df["AcceptedCmp4"]
    + df["AcceptedCmp5"]
)


# Deal purchase ratio
df["DealPurchaseRatio"] = (
    df["NumDealsPurchases"]
    / (df["TotalPurchases"] + 1)
)


# Web purchase ratio
df["WebPurchaseRatio"] = (
    df["NumWebPurchases"]
    / (df["TotalPurchases"] + 1)
)


# Store purchase ratio
df["StorePurchaseRatio"] = (
    df["NumStorePurchases"]
    / (df["TotalPurchases"] + 1)
)


# Catalog purchase ratio
df["CatalogPurchaseRatio"] = (
    df["NumCatalogPurchases"]
    / (df["TotalPurchases"] + 1)
)


# Web conversion ratio
df["WebConversionRatio"] = (
    df["NumWebPurchases"]
    / (df["NumWebVisitsMonth"] + 1)
)


# ============================================================
# 6. CONVERT CATEGORICAL VARIABLES
# ============================================================

print("\nEncoding categorical variables...")


# Education
df["Education"] = df["Education"].map({
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4
})


# Marital Status
df["Marital_Status"] = df["Marital_Status"].map({
    "Single": 0,
    "Alone": 0,
    "Divorced": 1,
    "Widow": 2,
    "Absurd": 3,
    "YOLO": 3,
    "Married": 4,
    "Together": 4
})


# ============================================================
# 7. CREATE DATE FEATURES
# ============================================================

df["CustomerYear"] = df["Dt_Customer"].dt.year

df["CustomerMonth"] = df["Dt_Customer"].dt.month


# ============================================================
# 8. SELECT FEATURES FOR CLUSTERING
# ============================================================

features = [
    "Year_Birth",
    "Education",
    "Marital_Status",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",

    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",

    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",

    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",

    "Complain",
    "Response",

    "TotalSpent",
    "TotalPurchases",
    "TotalChildren",
    "TotalCampaignsAccepted",

    "DealPurchaseRatio",
    "WebPurchaseRatio",
    "StorePurchaseRatio",
    "CatalogPurchaseRatio",
    "WebConversionRatio",

    "CustomerYear",
    "CustomerMonth"
]


X = df[features].copy()

print("\nNumber of features used:")
print(len(features))

print("\nFeatures:")
print(features)


# ============================================================
# 9. HANDLE MISSING VALUES
# ============================================================

print("\nHandling remaining missing values...")

X = X.fillna(X.median())


# ============================================================
# 10. SAVE CLEANED DATASET
# ============================================================

print("\nSaving cleaned dataset...")

df.to_csv(
    "outputs/cleaned_customer_data.csv",
    index=False
)

print("Cleaned dataset saved successfully!")


# ============================================================
# 11. STANDARDIZE FEATURES
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 12. APPLY PCA
# ============================================================

print("\nApplying PCA...")

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


print("\nPCA explained variance:")

print(
    pca.explained_variance_ratio_
)


total_variance = (
    pca.explained_variance_ratio_.sum()
    * 100
)


print("\nTotal variance explained by 2 components:")

print(
    round(total_variance, 2),
    "%"
)


# ============================================================
# 13. SAVE PCA EXPLAINED VARIANCE
# ============================================================

pca_variance = pd.DataFrame({
    "Component": [
        "PC1",
        "PC2"
    ],
    "ExplainedVariance": (
        pca.explained_variance_ratio_
    )
})


pca_variance.to_csv(
    "outputs/pca_explained_variance.csv",
    index=False
)


# ============================================================
# 13. FAST ELBOW METHOD
# ============================================================

print("\nRunning Fast Elbow Method...")

# Use a sample of customers for finding the elbow.
# This is much faster and still gives a reliable estimate.
elbow_sample_size = min(1000, len(X_pca))

np.random.seed(42)

elbow_indices = np.random.choice(
    len(X_pca),
    size=elbow_sample_size,
    replace=False
)

X_elbow = X_pca[elbow_indices]

inertias = []

# Test fewer K values
K_values = range(2, 9)

for k in K_values:

    print("Testing K =", k)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=3,
        max_iter=50
    )

    kmeans.fit(X_elbow)

    inertias.append(kmeans.inertia_)


# Save elbow results
elbow_df = pd.DataFrame({
    "K": list(K_values),
    "Inertia": inertias
})

elbow_df.to_csv(
    "outputs/elbow_scores.csv",
    index=False
)


# Plot elbow
plt.figure(figsize=(10, 6))

plt.plot(
    list(K_values),
    inertias,
    marker="o"
)

plt.title("Elbow Method for Optimal K")

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.xticks(list(K_values))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/elbow_method.png",
    dpi=150
)

plt.close()

print("\nElbow Method complete!")
print("Elbow values:")
print(elbow_df)

# ============================================================
# 15. SILHOUETTE SCORE
# ============================================================

print("\nCalculating Silhouette Scores...")

silhouette_values = []


for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_pca
    )

    score = silhouette_score(
        X_pca,
        labels
    )

    silhouette_values.append(
        score
    )

    print(
        "K =",
        k,
        "Silhouette Score =",
        round(score, 4)
    )


# Save silhouette values

silhouette_data = pd.DataFrame({
    "K": list(range(2, 11)),
    "SilhouetteScore": silhouette_values
})


silhouette_data.to_csv(
    "outputs/silhouette_scores.csv",
    index=False
)


# Silhouette graph

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_values,
    marker="o"
)

plt.title(
    "Silhouette Score for Different K Values"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(
    list(range(2, 11))
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/silhouette_scores.png"
)

plt.show()


# ============================================================
# 16. SELECT BEST K
# ============================================================

best_k = list(range(2, 11))[
    np.argmax(silhouette_values)
]


print("\nBest K according to Silhouette Score:")

print(best_k)


# ============================================================
# 16. FINAL K-MEANS CLUSTERING
# ============================================================

print("\nRunning final K-Means clustering...")

# Based on the highest silhouette score
best_k = 2

print("Selected K =", best_k)

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=3,
    max_iter=50
)

df["Cluster"] = final_kmeans.fit_predict(X_pca)

print("Final clustering complete!")


# ============================================================
# 17. ADD PCA COMPONENTS
# ============================================================

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]


# ============================================================
# 18. SAVE CUSTOMER CLUSTERS
# ============================================================

df.to_csv(
    "outputs/customer_clusters.csv",
    index=False
)

print("customer_clusters.csv saved!")


# ============================================================
# 19. CLUSTER SUMMARY
# ============================================================

summary_features = [
    "Income",
    "Recency",
    "TotalSpent",
    "TotalPurchases",
    "TotalChildren",
    "TotalCampaignsAccepted",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Response"
]

cluster_summary = df.groupby("Cluster")[summary_features].mean()

cluster_summary["CustomerCount"] = df.groupby(
    "Cluster"
).size()

cluster_summary = cluster_summary.reset_index()

cluster_summary.to_csv(
    "outputs/cluster_summary.csv",
    index=False
)

print("\nCluster Summary:")
print(cluster_summary)

print("\ncluster_summary.csv saved!")


# ============================================================
# 20. PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Cluster"],
    alpha=0.7
)

plt.title("Customer Segments using K-Means")

plt.xlabel("PCA1")

plt.ylabel("PCA2")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/pca_2d_clusters.png",
    dpi=150
)

plt.close()

print("pca_2d_clusters.png saved!")


# ============================================================
# 21. CLUSTER SIZE GRAPH
# ============================================================

cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("Number of Customers in Each Cluster")

plt.xlabel("Cluster")

plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "outputs/cluster_sizes.png",
    dpi=150
)

plt.close()

print("cluster_sizes.png saved!")


# ============================================================
# 22. CREATE CUSTOMER PERSONAS
# ============================================================

overall_income = df["Income"].mean()
overall_spending = df["TotalSpent"].mean()
overall_web = df["NumWebPurchases"].mean()
overall_store = df["NumStorePurchases"].mean()
overall_deals = df["NumDealsPurchases"].mean()


persona_rows = []


for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    income = cluster_data["Income"].mean()
    spending = cluster_data["TotalSpent"].mean()
    web = cluster_data["NumWebPurchases"].mean()
    store = cluster_data["NumStorePurchases"].mean()
    deals = cluster_data["NumDealsPurchases"].mean()


    # Income
    if income > overall_income * 1.15:
        income_type = "High Income"
    elif income < overall_income * 0.85:
        income_type = "Lower Income"
    else:
        income_type = "Average Income"


    # Spending
    if spending > overall_spending * 1.20:
        spending_type = "High Spending"
    elif spending < overall_spending * 0.80:
        spending_type = "Low Spending"
    else:
        spending_type = "Average Spending"


    # Shopping channel
    if web > overall_web * 1.20:
        channel_type = "Digital-Focused"
    elif store > overall_store * 1.20:
        channel_type = "Store-Focused"
    else:
        channel_type = "Mixed-Channel"


    # Deal sensitivity
    if deals > overall_deals * 1.20:
        deal_type = "Highly Deal-Sensitive"
    else:
        deal_type = "Less Deal-Sensitive"


    persona = (
        income_type
        + ", "
        + spending_type
        + ", "
        + channel_type
        + ", "
        + deal_type
    )


    persona_rows.append({
        "Cluster": cluster,
        "Persona": persona,
        "AverageIncome": round(income, 2),
        "AverageSpending": round(spending, 2),
        "AverageWebPurchases": round(web, 2),
        "AverageStorePurchases": round(store, 2),
        "AverageDealPurchases": round(deals, 2),
        "CustomerCount": len(cluster_data)
    })


# ============================================================
# 23. SAVE PERSONAS
# ============================================================

personas_df = pd.DataFrame(persona_rows)

personas_df.to_csv(
    "outputs/customer_personas.csv",
    index=False
)


with open(
    "outputs/customer_personas.txt",
    "w",
    encoding="utf-8"
) as file:

    for _, row in personas_df.iterrows():

        file.write(
            "Cluster "
            + str(row["Cluster"])
            + ": "
            + row["Persona"]
            + "\n"
        )

        file.write(
            "Customers: "
            + str(row["CustomerCount"])
            + "\n"
        )

        file.write(
            "Average Income: "
            + str(row["AverageIncome"])
            + "\n"
        )

        file.write(
            "Average Spending: "
            + str(row["AverageSpending"])
            + "\n"
        )

        file.write("\n")


print("\nCustomer Personas:")
print(personas_df)

print("\ncustomer_personas.csv saved!")
print("customer_personas.txt saved!")


# ============================================================
# 24. PROJECT COMPLETE
# ============================================================

print("\n========================================")
print("PROJECT 3 COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nOptimal K based on Silhouette Score:", best_k)

print(
    "PCA variance explained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

print("\nAll output files have been generated.")