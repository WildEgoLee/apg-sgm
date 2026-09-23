import csv
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else 'results/ablation_middlebury.csv'

with open(filename) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

by_case = {}
for r in rows:
    by_case.setdefault(r['case'], {})[r['ablation_id']] = r

print(f"Total scenes: {len(by_case)}")
header = f"{'Scene':<14} {'Rec(vis)':<9} {'Rec(mat)':<9} {'P@1.0':<8} {'GridCov':<9} {'SuppMAE':<9} {'PriorRed':<9} {'D Bad2':<8} {'E Bad2':<8} {'dBad2':<8} {'dEPE':<8}"
print(header)
print("-" * len(header))

rec_vis_list = []
rec_mat_list = []
prior_red_list = []
delta_bad2_list = []
delta_epe_list = []

tot_vis_eval = 0
tot_vis_in = 0
tot_mat_eval = 0
tot_mat_in = 0

supp_p05_list = []
supp_p1_list = []
supp_p2_list = []
supp_mae_list = []
supp_cov_list = []

for case, abs_dict in by_case.items():
    d = abs_dict['D']
    e = abs_dict['E']
    rec_vis = float(e['recall_visible']) * 100
    rec_mat = float(e['recall_matchable']) * 100
    p_red = float(e['prior_incremental_reduction']) * 100
    d_bad2 = float(d['bad_2_0'])
    e_bad2 = float(e['bad_2_0'])
    d_epe = float(d['epe'])
    e_epe = float(e['epe'])
    del_bad2 = e_bad2 - d_bad2
    del_epe = e_epe - d_epe
    
    rec_vis_list.append(rec_vis)
    rec_mat_list.append(rec_mat)
    prior_red_list.append(p_red)
    delta_bad2_list.append(del_bad2)
    delta_epe_list.append(del_epe)

    # Pixel counts for micro recall
    vis_eval = int(e.get('range_vis_eval_px', 0))
    vis_in = int(e.get('range_vis_in_px', 0))
    mat_eval = int(e.get('range_mat_eval_px', 0))
    mat_in = int(e.get('range_mat_in_px', 0))

    tot_vis_eval += vis_eval
    tot_vis_in += vis_in
    tot_mat_eval += mat_eval
    tot_mat_in += mat_in

    # Support metrics
    p05 = float(e.get('support_p05', 0)) * 100
    p1 = float(e.get('support_p1', 0)) * 100
    p2 = float(e.get('support_p2', 0)) * 100
    mae = float(e.get('support_mae', 0))
    cov = float(e.get('grid_coverage', 0)) * 100

    supp_p05_list.append(p05)
    supp_p1_list.append(p1)
    supp_p2_list.append(p2)
    supp_mae_list.append(mae)
    supp_cov_list.append(cov)
    
    print(f"{case:<14} {rec_vis:5.2f}%   {rec_mat:5.2f}%   {p1:5.1f}%   {cov:5.1f}%    {mae:5.2f}px   {p_red:5.1f}%    {d_bad2:5.2f}%   {e_bad2:5.2f}%   {del_bad2:+5.2f}%  {del_epe:+6.3f}")

print("-" * len(header))
micro_vis = (tot_vis_in / tot_vis_eval * 100.0) if tot_vis_eval > 0 else 0.0
micro_mat = (tot_mat_in / tot_mat_eval * 100.0) if tot_mat_eval > 0 else 0.0

print(f"Macro Visible Recall: {sum(rec_vis_list)/len(rec_vis_list):.2f}%")
print(f"Micro Visible Recall: {micro_vis:.2f}% (Total: {tot_vis_in}/{tot_vis_eval})")
print(f"Worst Scene Recall:   {min(rec_vis_list):.2f}%")
print(f"Macro Matchable Rec:  {sum(rec_mat_list)/len(rec_mat_list):.2f}% (Micro: {micro_mat:.2f}%)")
print(f"Support Precision@1:  {sum(supp_p1_list)/len(supp_p1_list):.2f}% (P@0.5: {sum(supp_p05_list)/len(supp_p05_list):.2f}%, P@2: {sum(supp_p2_list)/len(supp_p2_list):.2f}%)")
print(f"Support Mean Abs Err: {sum(supp_mae_list)/len(supp_mae_list):.3f} px")
print(f"Support Grid Coverage:{sum(supp_cov_list)/len(supp_cov_list):.2f}%")
print(f"Prior Space Reduction:{sum(prior_red_list)/len(prior_red_list):.2f}%")
print(f"Delta Bad-2 (D -> E): {sum(delta_bad2_list)/len(delta_bad2_list):+.2f}% (Max increase: {max(delta_bad2_list):+.2f}%)")
print(f"Delta EPE   (D -> E): {sum(delta_epe_list)/len(delta_epe_list):+.3f} px")
