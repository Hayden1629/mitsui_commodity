import pandas as pd
import numpy as np

train = pd.read_csv('/Users/hayden/coderepos_mac_mini/mitsui_commodity/data/train.csv')
labels = pd.read_csv('/Users/hayden/coderepos_mac_mini/mitsui_commodity/data/train_labels.csv')
pairs = pd.read_csv('/Users/hayden/coderepos_mac_mini/mitsui_commodity/data/target_pairs.csv')

# parse pairs as before
def parse_pair(pair_str):
    if ' - ' in pair_str:
        a, b = pair_str.split(' - ')
        return a.strip(), b.strip()
    return pair_str.strip(), None

pairs[['asset_a', 'asset_b']] = pairs['pair'].apply(lambda x: pd.Series(parse_pair(x)))

# set date_id as index for alignment
train = train.set_index('date_id')
labels = labels.set_index('date_id')

# pick one target to test your assumption
row = pairs.iloc[2]  # target_2: LME_CA_Close - LME_ZS_Close
print(f"Testing: {row['pair']}")

a = train[row['asset_a']]
b = train[row['asset_b']]

# hypothesis 1: log return of A minus log return of B
log_ret_a = np.log(a).diff()
log_ret_b = np.log(b).diff()
reconstructed_log = (log_ret_a - log_ret_b).shift(-2)  # shift forward by lag+1

# hypothesis 2: simple return difference
simple_ret_a = a.pct_change()
simple_ret_b = b.pct_change()
reconstructed_simple = (simple_ret_a - simple_ret_b).shift(-2)  # shift forward by lag+1

# compare against actual label
actual = labels[row['target']]

print("\nActual label (first 5):")
print(actual.head())
print("\nReconstructed log return diff:")
print(reconstructed_log.head())
print("\nReconstructed simple return diff:")
print(reconstructed_simple.head())

# check correlation to see which is closer
corr_log = actual.corr(reconstructed_log)
corr_simple = actual.corr(reconstructed_simple)
print(f"\nCorrelation (log): {corr_log:.6f}")
if corr_log > 0.99:
    print("Huzzah!")
    print("Log return difference is a better match.")
print(f"Correlation (simple): {corr_simple:.6f}")

# supra I have demonstrated that the train labels are indeed the log return differences, lagged by 2 days. This means that the target for date_id t is log_return(asset_a at t-2) - log_return(asset_b at t-2).
# This is a crucial insight for how to approach modeling, as it confirms that we are predicting a lagged log return difference, which has implications for feature engineering and model design.

#now I want to do this with the test data, and verify that is it the same as what ethan came up with, test_ground_truth.csv
test_ground_truth = pd.read_csv('/Users/hayden/coderepos_mac_mini/mitsui_commodity/data/test_ground_truth.csv')
test_ground_truth = test_ground_truth.set_index('date_id')

#target file to be made is test labels and compare to test_ground_truth
# load test data
test = pd.read_csv('/Users/hayden/coderepos_mac_mini/mitsui_commodity/data/test.csv')
test = test.set_index('date_id')

# reconstruct labels for all pairs using the confirmed formula:
# label_t = log_return(asset_a, t-2) - log_return(asset_b, t-2)
# which in shifted terms = (log_ret_a - log_ret_b).shift(-2)

reconstructed_test_labels = {}

for _, row in pairs.iterrows():
    a = test[row['asset_a']]
    
    if pd.notna(row['asset_b']):
        b = test[row['asset_b']]
        log_ret_diff = np.log(a).diff() - np.log(b).diff()
    else:
        # single asset target
        log_ret_diff = np.log(a).diff()
    
    reconstructed_test_labels[row['target']] = log_ret_diff.shift(-2)

reconstructed_df = pd.DataFrame(reconstructed_test_labels)


print("===============Test data checking =====================")
# compare against ground truth
print("Ground truth shape:", test_ground_truth.shape)
print("Reconstructed shape:", reconstructed_df.shape)

# align on common index
common_idx = test_ground_truth.index.intersection(reconstructed_df.index)
gt_aligned = test_ground_truth.loc[common_idx]
rc_aligned = reconstructed_df.loc[common_idx]

print("\nPer-target correlation:")
for col in gt_aligned.columns:
    if col in rc_aligned.columns:
        corr = gt_aligned[col].corr(rc_aligned[col])
        match = "✓" if corr > 0.99 else "✗"
        print(f"  {match} {col}: {corr:.6f}")

# overall summary
correlations = [gt_aligned[col].corr(rc_aligned[col]) 
                for col in gt_aligned.columns if col in rc_aligned.columns]
print(f"\nMean correlation: {np.mean(correlations):.6f}")
print(f"All > 0.99: {all(c > 0.99 for c in correlations)}")