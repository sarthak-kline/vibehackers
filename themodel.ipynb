from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from tqdm import tqdm
from rapidfuzz.fuzz import token_set_ratio

# === Load CSV ===
df = pd.read_csv('FinalData.csv')

# === Drop rows with missing SourceDescription or ProductName ===
df = df.dropna(subset=['SourceDescription', 'ProductName'])

# === Fill NaNs in brand columns ===
for col in ['SourceBrand', 'SourceSubBrand', 'SourceMasterBrand', 'BrandName', 'SubBrandName', 'MasterBrandName']:
    df[col] = df[col].fillna('')

# === Function to enrich with brand info ===
def enrich_with_brand(desc, brand, subbrand, masterbrand):
    parts = [brand, subbrand, masterbrand]
    brand_str = ' '.join([p.strip() for p in parts if p.strip()])
    return f"{brand_str} {desc}".strip()

# === Combined Source Text ===
df['CombinedSourceText'] = df.apply(
    lambda row: f"Description: {enrich_with_brand(row['SourceDescription'], row['SourceBrand'], row['SourceSubBrand'], row['SourceMasterBrand'])}",
    axis=1
)

# === Prepare Master Data ===
master_df = df[['ProductName', 'BrandName', 'SubBrandName', 'MasterBrandName']].drop_duplicates().fillna('')
master_df['CombinedMasterText'] = master_df.apply(
    lambda row: f"Description: {enrich_with_brand(row['ProductName'], row['BrandName'], row['SubBrandName'], row['MasterBrandName'])}",
    axis=1
)

# === Load SBERT Model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Encode Master Texts ===
master_texts = master_df['CombinedMasterText'].tolist()
master_embeddings = model.encode(master_texts, convert_to_tensor=True, show_progress_bar=True)

# === Batched Matching ===
batch_size = 5000
source_texts = df['CombinedSourceText'].tolist()

best_indices = []
best_scores = []

for i in tqdm(range(0, len(source_texts), batch_size), desc="Matching"):
    batch_texts = source_texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
    
    scores = util.cos_sim(batch_embeddings, master_embeddings)  # [batch_size x M]
    best_batch_scores, best_batch_indices = scores.max(dim=1)

    best_scores.extend(best_batch_scores.cpu().numpy())
    best_indices.extend(best_batch_indices.cpu().numpy())

# === Add Matching Results ===
df = df.iloc[:len(best_scores)].copy()  # ensure alignment

df['BestMatchProduct'] = [master_df.iloc[i]['ProductName'] for i in best_indices]
df['MatchedBrand'] = [master_df.iloc[i]['BrandName'] for i in best_indices]
df['MatchedSubBrand'] = [master_df.iloc[i]['SubBrandName'] for i in best_indices]
df['MatchedMasterBrand'] = [master_df.iloc[i]['MasterBrandName'] for i in best_indices]

# === Score Calculation ===

# SBERT score
df['SBERTScore'] = best_scores

# Fuzzy match (ProductNameScore)
def fuzzy_overlap(a, b):
    return token_set_ratio(str(a), str(b)) / 100.0

df['ProductNameScore'] = df.apply(
    lambda row: fuzzy_overlap(row['SourceDescription'], row['BestMatchProduct']),
    axis=1
)

# Combine scores using alpha weight
alpha = 0.8  # More weight to SBERT for precision
df['SimilarityScore'] = df.apply(
    lambda row: alpha * row['SBERTScore'] + (1 - alpha) * row['ProductNameScore'],
    axis=1
)

# === Adjust similarity based on fuzzy score ===
def adjust_similarity(row):
    sim = row['SimilarityScore']
    fuzzy = row['ProductNameScore']
    
    if fuzzy >= 0.90:
        return min(sim + 0.50, 1.0)
    elif fuzzy <= 0.30:
        return max(sim - 0.03, 0.0)
    else:
        return sim

df['AdjustedSimilarityScore'] = df.apply(adjust_similarity, axis=1)

# === Confidence tagging based on adjusted score ===
threshold = 0.65
df['Confidence'] = df['AdjustedSimilarityScore'].apply(lambda x: 'High' if x >= threshold else 'Low')

# === Save Output ===
# Select only the desired output columns
output_df = df[['ServiceAndProductMappingId', 'ProductMasterId', 'ProductName', 'BestMatchProduct', 'AdjustedSimilarityScore', 'Confidence']]

# Save to CSV
output_df.to_csv('FinalOutput.csv', index=False)

print("✅ Matching complete! Filtered results saved to 'FinalOutput.csv'")
