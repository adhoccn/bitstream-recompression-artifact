"""
Bitstream feature classifier evaluation.
- 5-fold CV with LR and XGBoost
- Exclude confounding features (duration, width, height)
- Per-codec breakdown
- Comparison with pixel-domain results
"""
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLIND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BLIND_DIR)
warnings.filterwarnings('ignore')

FEATURES_CSV = os.path.join(BLIND_DIR, 'results', 'bitstream_features_full.csv')
PIXEL_CSV = os.path.join(BLIND_DIR, 'results', 'pipeline_results_full.csv')

# Features to exclude (confounding, metadata, non-numeric)
EXCLUDE = {'duration_s', 'width', 'height', 'video_file', 'label', 'codec',
           'crf', 'sample_path', 'codec_name', 'profile', 'pixel_format',
           'error', 'n_frames_decoded', 'other_ratio', 'level'}

def load_features(csv_path):
    """Load features from CSV, return X, y, feature_names."""
    rows = list(csv.DictReader(open(csv_path, 'r', encoding='utf-8')))
    
    # Identify numeric feature columns
    feature_names = []
    for key in rows[0].keys():
        if key in EXCLUDE:
            continue
        try:
            float(rows[0][key])
            feature_names.append(key)
        except (ValueError, TypeError):
            continue
    
    # Build X matrix and y labels
    X = []
    y = []
    labels_list = []
    codecs = []
    
    for row in rows:
        label = row.get('label', '')
        if label not in ('single', 'double'):
            continue
        
        vals = []
        valid = True
        for fn in feature_names:
            try:
                v = float(row.get(fn, 0))
                if np.isnan(v) or np.isinf(v):
                    v = 0
                vals.append(v)
            except (ValueError, TypeError):
                vals.append(0)
        
        X.append(vals)
        y.append(1 if label == 'double' else 0)
        labels_list.append(label)
        codecs.append(row.get('codec', ''))
    
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    
    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    return X, y, feature_names, np.array(codecs)


def evaluate_cv(X, y, feature_names, codecs, n_folds=5, verbose=True):
    """5-fold stratified CV evaluation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    lr_aucs = []
    lr_f1s = []
    lr_fprs = []
    lr_accs = []
    
    # Try XGBoost
    xgb_available = True
    try:
        from xgboost import XGBClassifier
    except ImportError:
        xgb_available = False
    
    xgb_aucs = []
    xgb_f1s = []
    xgb_fprs = []
    
    print(f"\n{'='*70}")
    print(f"BITSTREAM FEATURE CLASSIFICATION: {n_folds}-FOLD CV")
    print(f"{'='*70}")
    print(f"Samples: {len(y)} (single={sum(y==0)}, double={sum(y==1)})")
    print(f"Features: {len(feature_names)}")
    print(f"XGBoost: {'available' if xgb_available else 'NOT available'}")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # LR
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(X_train_s, y_train)
        y_prob = lr.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        # FPR: FP / (FP + TN)
        fp = sum((y_pred == 1) & (y_test == 0))
        tn = sum((y_pred == 0) & (y_test == 0))
        fpr = fp / (fp + tn + 1e-10)
        
        lr_aucs.append(auc)
        lr_f1s.append(f1)
        lr_fprs.append(fpr)
        lr_accs.append(acc)
        
        if verbose:
            print(f"  Fold {fold+1}: LR AUC={auc:.4f}, F1={f1:.4f}, FPR={fpr:.4f}, Acc={acc:.4f}")
        
        # XGBoost
        if xgb_available:
            xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                random_state=42, eval_metric='logloss', verbosity=0)
            xgb.fit(X_train, y_train)
            y_prob_x = xgb.predict_proba(X_test)[:, 1]
            y_pred_x = (y_prob_x >= 0.5).astype(int)
            
            auc_x = roc_auc_score(y_test, y_prob_x)
            f1_x = f1_score(y_test, y_pred_x)
            fp_x = sum((y_pred_x == 1) & (y_test == 0))
            tn_x = sum((y_pred_x == 0) & (y_test == 0))
            fpr_x = fp_x / (fp_x + tn_x + 1e-10)
            
            xgb_aucs.append(auc_x)
            xgb_f1s.append(f1_x)
            xgb_fprs.append(fpr_x)
    
    print(f"\n--- LR Summary ---")
    print(f"  AUC: {np.mean(lr_aucs):.4f} +/- {np.std(lr_aucs):.4f}")
    print(f"  F1:  {np.mean(lr_f1s):.4f} +/- {np.std(lr_f1s):.4f}")
    print(f"  FPR: {np.mean(lr_fprs):.4f} +/- {np.std(lr_fprs):.4f}")
    print(f"  Acc: {np.mean(lr_accs):.4f} +/- {np.std(lr_accs):.4f}")
    
    if xgb_available and xgb_aucs:
        print(f"\n--- XGBoost Summary ---")
        print(f"  AUC: {np.mean(xgb_aucs):.4f} +/- {np.std(xgb_aucs):.4f}")
        print(f"  F1:  {np.mean(xgb_f1s):.4f} +/- {np.std(xgb_f1s):.4f}")
        print(f"  FPR: {np.mean(xgb_fprs):.4f} +/- {np.std(xgb_fprs):.4f}")
    
    # Feature importance from LR
    print(f"\n--- LR Feature Importance (|coef|) ---")
    # Train final model on all data
    scaler_all = StandardScaler()
    X_all_s = scaler_all.fit_transform(X)
    lr_final = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_final.fit(X_all_s, y)
    
    coefs = lr_final.coef_[0]
    importance = sorted(zip(feature_names, np.abs(coefs)), key=lambda x: x[1], reverse=True)
    print(f"  {'Feature':>25s} {'|coef|':>8s} {'coef':>8s}")
    for fn, abs_c, c in [(fn, abs_c, coefs[feature_names.index(fn)]) for fn, abs_c in importance[:15]]:
        print(f"  {fn:>25s} {abs_c:8.4f} {c:8.4f}")
    
    if xgb_available:
        print(f"\n--- XGBoost Feature Importance ---")
        xgb_final = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                   random_state=42, eval_metric='logloss', verbosity=0)
        xgb_final.fit(X, y)
        imp = xgb_final.feature_importances_
        importance_xgb = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
        for fn, im in importance_xgb[:15]:
            print(f"  {fn:>25s} {im:8.4f}")
    
    # Per-codec breakdown
    print(f"\n{'='*70}")
    print("PER-CODEC AUC BREAKDOWN (LR, full data)")
    print(f"{'='*70}")
    
    y_prob_all = lr_final.predict_proba(X_all_s)[:, 1]
    overall_auc = roc_auc_score(y, y_prob_all)
    print(f"  Overall AUC: {overall_auc:.4f}")
    
    for codec in ['h264', 'h265', 'vp9']:
        mask_double = codecs == codec
        mask_single = codecs == 'original'
        
        if mask_double.sum() == 0:
            continue
        
        # Single vs this codec's double
        combined_mask = mask_single | mask_double
        y_sub = y[combined_mask]
        prob_sub = y_prob_all[combined_mask]
        
        if len(np.unique(y_sub)) < 2:
            continue
        
        auc_sub = roc_auc_score(y_sub, prob_sub)
        print(f"  Single vs {codec}: AUC={auc_sub:.4f} (N={combined_mask.sum()})")
    
    return lr_aucs, lr_f1s


def compare_with_pixel_domain():
    """Compare bitstream results with pixel-domain results."""
    print(f"\n{'='*70}")
    print("COMPARISON: PIXEL-DOMAIN vs BITSTREAM-DOMAIN")
    print(f"{'='*70}")
    
    pixel_rows = list(csv.DictReader(open(PIXEL_CSV, 'r', encoding='utf-8')))
    
    # Pixel domain had: F1=0.861, FPR=0.900, all stages AUC ~0.5
    print(f"\n  {'Metric':>15s} {'Pixel-Domain':>15s} {'Bitstream':>15s} {'Improvement':>12s}")
    print(f"  {'-'*60}")
    
    # Pixel domain stats
    pixel_scores = [float(r['final_score']) for r in pixel_rows]
    pixel_labels = [1 if r['label'] == 'double' else 0 for r in pixel_rows]
    pixel_single = [s for s, l in zip(pixel_scores, pixel_labels) if l == 0]
    pixel_double = [s for s, l in zip(pixel_scores, pixel_labels) if l == 1]
    
    pixel_auc = roc_auc_score(pixel_labels, pixel_scores)
    pixel_preds = [1 if s >= 0.6032 else 0 for s in pixel_scores]  # T1 threshold
    pixel_f1 = f1_score(pixel_labels, pixel_preds)
    fp = sum((p == 1) & (l == 0) for p, l in zip(pixel_preds, pixel_labels))
    tn = sum((p == 0) & (l == 0) for p, l in zip(pixel_preds, pixel_labels))
    pixel_fpr = fp / (fp + tn + 1e-10)
    
    print(f"  {'AUC':>15s} {pixel_auc:>15.4f} {'see above':>15s}")
    print(f"  {'F1':>15s} {pixel_f1:>15.4f} {'see above':>15s}")
    print(f"  {'FPR':>15s} {pixel_fpr:>15.4f} {'see above':>15s}")
    
    # Pixel domain stage AUCs
    for stage in ['stage1_score', 'stage2_score', 'stage3_score', 'stage4_score']:
        vals = [float(r[stage]) for r in pixel_rows]
        if len(set(vals)) > 1:
            auc = roc_auc_score(pixel_labels, vals)
            print(f"  {stage:>15s} AUC={auc:.4f}")
        else:
            print(f"  {stage:>15s} AUC=N/A (constant)")


def main():
    print("Loading bitstream features...")
    X, y, feature_names, codecs = load_features(FEATURES_CSV)
    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    
    # Evaluate
    evaluate_cv(X, y, feature_names, codecs)
    
    # Compare with pixel domain
    compare_with_pixel_domain()
    
    # Baseline: "always predict double"
    print(f"\n{'='*70}")
    print("BASELINE: ALWAYS PREDICT DOUBLE")
    print(f"{'='*70}")
    n_double = sum(y == 1)
    n_total = len(y)
    baseline_f1 = 2 * n_double / (n_total + n_double)
    baseline_acc = n_double / n_total
    print(f"  F1={baseline_f1:.4f}, Acc={baseline_acc:.4f}, FPR=1.0000")


if __name__ == '__main__':
    main()
